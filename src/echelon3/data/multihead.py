"""Multi-binary-head segmentation dataset.

Layout expected on disk:
    <image_folder>/.../<name>.<image_ext>
    <heads_root>/<head_name>/.../<name>.<mask_ext>     # uint8 {0, 1, 255}

If a head's mask file does not exist for a given image, that head receives an
all-`ignore_index` mask for the sample (loss/metric will skip these pixels).

Returns per sample: (image_tensor, {head_name: mask_tensor_long_HxW}).
"""
from __future__ import annotations

import glob
import os
from random import Random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Note: cv2.setNumThreads(0) and BLAS thread caps are applied package-wide in
# echelon3/data/__init__.py — applies to every dataset, not just this one.


class MultiBinarySegDataset(Dataset):

    def __init__(
        self,
        image_folder: str,
        image_extension: str,
        heads_root: str,
        head_names: List[str],
        mask_extension: str = "png",
        augment=None,
        preprocess=None,
        ignore_index: int = 255,
        max_files: int | None = None,
        seed: int = 0,
        file_list_txt: str | None = None,
        image_glob: str | None = None,
        flat_mask_lookup: bool = False,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.image_extension = image_extension
        self.heads_root = heads_root
        self.head_names = list(head_names)
        self.mask_extension = mask_extension
        self.augment = augment
        self.preprocess = preprocess
        self.ignore_index = int(ignore_index)
        self.flat_mask_lookup = bool(flat_mask_lookup)

        if file_list_txt is not None:
            with open(file_list_txt) as f:
                stems = [ln.strip() for ln in f if ln.strip()]
            # Try direct join first; fall back to recursive basename lookup for
            # nested layouts (e.g. OpenEarthMap where entry "aachen_1.tif"
            # lives at <region>/images/<entry>).
            paths = []
            by_name: dict[str, str] | None = None
            for s in stems:
                p = os.path.join(image_folder, s)
                if os.path.isfile(p):
                    paths.append(p); continue
                p2 = os.path.join(image_folder, os.path.splitext(s)[0] + "." + image_extension)
                if os.path.isfile(p2):
                    paths.append(p2); continue
                if by_name is None:
                    by_name = {
                        os.path.basename(cp): cp for cp in glob.glob(
                            os.path.join(image_folder, "**", "*." + image_extension),
                            recursive=True,
                        )
                    }
                key = s if "." in s else os.path.splitext(s)[0] + "." + image_extension
                if key in by_name:
                    paths.append(by_name[key])
                else:
                    raise FileNotFoundError(
                        f"file_list entry {s!r} not found under {image_folder}"
                    )
        elif image_glob is not None:
            # explicit relative glob (e.g. '*/images/*.tif' for OEM-style layouts)
            paths = sorted(glob.glob(os.path.join(image_folder, image_glob)))
        else:
            paths = sorted(glob.glob(
                os.path.join(image_folder, "**", "*." + image_extension),
                recursive=True,
            ))

        if shuffle:
            Random(seed).shuffle(paths)
        if max_files is not None:
            paths = paths[: int(max_files)]
        self.image_paths = paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __str__(self) -> str:
        return f"{type(self).__name__}(length={len(self)}, heads={self.head_names})"

    def _mask_path(self, img_path: str, head: str) -> str:
        if self.flat_mask_lookup:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            return os.path.join(self.heads_root, head, stem + "." + self.mask_extension)
        rel = os.path.relpath(img_path, self.image_folder)
        rel = os.path.splitext(rel)[0] + "." + self.mask_extension
        return os.path.join(self.heads_root, head, rel)

    def __getitem__(self, idx: int):
        # Resilient to single bad files: on any read failure we cycle to the
        # next index instead of crashing the worker (the latter caused a
        # PyTorch DataLoader deadlock with persistent_workers=True in v3).
        try:
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path)
            if image is None:
                raise RuntimeError(f"failed to read image {img_path}")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.image_paths))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        masks_in_order: List[np.ndarray] = []
        for head in self.head_names:
            mp = self._mask_path(img_path, head)
            if os.path.exists(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is None or m.shape[:2] != (h, w):
                    if m is not None:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        m = np.full((h, w), self.ignore_index, dtype=np.uint8)
            else:
                m = np.full((h, w), self.ignore_index, dtype=np.uint8)
            masks_in_order.append(m)

        if self.augment is not None:
            tr = self.augment(image=image, masks=masks_in_order)
            image_t = tr["image"]
            masks_t = tr["masks"]
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1)
            masks_t = [torch.from_numpy(m) for m in masks_in_order]

        if self.preprocess is not None:
            image_t = self.preprocess(image_t)

        labels: Dict[str, torch.Tensor] = {}
        for name, m in zip(self.head_names, masks_t):
            if not torch.is_tensor(m):
                m = torch.from_numpy(np.asarray(m))
            labels[name] = m.long()

        return image_t, labels

    def get_source_path(self, idx: int) -> str | None:
        if 0 <= idx < len(self.image_paths):
            return self.image_paths[idx]
        return None


def multihead_collate(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """Default collate compatible with dict labels."""
    images = torch.stack([b[0] for b in batch], dim=0)
    head_names = list(batch[0][1].keys())
    labels: Dict[str, torch.Tensor] = {
        h: torch.stack([b[1][h] for b in batch], dim=0) for h in head_names
    }
    return images, labels


class MultiBinarySegConcatDataset(Dataset):
    """Concat several MultiBinarySegDataset-compatible subsets.

    Used so a single train_dataset config can mix LandCover.ai + LoveDA + DeepGlobe ...
    Each `parts[i]` is a full `{module, type, config}` triple resolved via
    `echelon3.creator.create_single_dataset` (so it builds the subdatasets with
    the shared augment+preprocess).
    """

    def __init__(self, parts, augment, preprocess, **kwargs):
        super().__init__()
        from echelon3.creator import create_single_dataset
        self.subs: List[Dataset] = []
        for p in parts:
            self.subs.append(create_single_dataset(p, augment=augment, preprocess=preprocess))
        self.cumul: List[int] = []
        s = 0
        for d in self.subs:
            s += len(d)
            self.cumul.append(s)

    def __len__(self) -> int:
        return self.cumul[-1] if self.cumul else 0

    def __str__(self) -> str:
        parts_repr = ", ".join(f"{type(s).__name__}({len(s)})" for s in self.subs)
        return f"{type(self).__name__}(total={len(self)}, parts=[{parts_repr}])"

    def __getitem__(self, idx: int):
        for i, c in enumerate(self.cumul):
            if idx < c:
                prev = self.cumul[i - 1] if i > 0 else 0
                return self.subs[i][idx - prev]
        raise IndexError(idx)
