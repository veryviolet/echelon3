import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import shutil  # added for copying source files
import cv2
import numpy as np

from echelon3.evaluators.basic import Evaluator
from echelon3.data.imageclassifier import FoldersHiveImageClassifierDataset


class ClassifierEvaluator(Evaluator):
    """Evaluate a classifier on the validation (test) dataset
    + save misclassified samples into per-class folders X inside scores_and_labels.

    Folder X holds the samples whose true class is X
    but which the model gets wrong.

    The evaluator.config must contain:
      - scores_and_labels: str  # base folder for saving errors
    """

    def __init__(
        self,
        net,
        dataloader: Optional[DataLoader],
        metric,
        preprocess,
        postprocess,
        scores_and_labels: str,
        **kwargs,
    ):
        super().__init__(
            net=net,
            dataloader=dataloader,
            metric=metric,
            preprocess=preprocess,
            postprocess=postprocess,
            **kwargs,
        )

        self.errors_root = scores_and_labels
        os.makedirs(self.errors_root, exist_ok=True)

        # move the metric to the same device as the network (if it supports to)
        if hasattr(self.metric, "to"):
            self.metric = self.metric.to(self.device)

    def _build_errors_dataset(
        self,
        error_classes: set[int],
        base_dataset,
    ) -> Optional[FoldersHiveImageClassifierDataset]:
        """
        Builds a dataset from the error folders:
          errors_root/<class_id>/*.png|*.jpg|...

        Uses the SAME augment and preprocess as the original test dataset,
        so the pipeline fully matches the main validator.
        """
        if not error_classes:
            return None

        num_classes = max(error_classes) + 1

        ds = FoldersHiveImageClassifierDataset(
            augment=getattr(base_dataset, "augment", None),
            preprocess=getattr(base_dataset, "preprocess", None),
            label_type="class",
            classes=num_classes,
            folder=self.errors_root,
            wildcards=["*.png", "*.jpg", "*.jpeg", "*.bmp"],
        )
        return ds

    def _evaluate_on_errors(self, batch_size: int, error_classes: set[int], base_dataset):
        """
        Extra pass: compute the metric on the collected errors.
        We expect accuracy (and similar metrics) to be 0
        (since we collected only misclassified examples).
        """
        errors_dataset = self._build_errors_dataset(error_classes, base_dataset)
        if errors_dataset is None or len(errors_dataset) == 0:
            print("\n--> No error samples collected, skipping secondary evaluation on errors.")
            return

        errors_loader = DataLoader(
            errors_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Reset the metric and run over the errors
        if hasattr(self.metric, "reset"):
            self.metric.reset()

        self.net.to(self.device)
        self.net.eval()

        with torch.no_grad():
            total_size = len(errors_dataset)
            progress = tqdm(
                initial=0,
                total=total_size,
                desc="--> Evaluating on collected errors",
                ncols=0,
            )

            for images, labels in errors_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.net(images)

                # update the metric
                self.metric.update(logits, labels)

                batch_size = images.size(0)
                progress.update(batch_size)

            progress.close()

        result = self.metric.compute()
        if torch.is_tensor(result) and result.numel() == 1:
            result = float(result.cpu())

        print(f"--> Metric on collected errors: {result}")
        return result

    def evaluate_one(self, dataloader: DataLoader, mode: str = "test"):
        if dataloader is None:
            raise RuntimeError("ClassifierEvaluator: dataloader is None")

        self.net.to(self.device)
        self.net.eval()

        if hasattr(self.metric, "reset"):
            self.metric.reset()

        # global index only for the fallback tensor saving
        global_idx = 0
        # check whether the dataset can return the path to the source file
        has_source_path = hasattr(dataloader.dataset, "get_source_path")
        # index offset within the whole dataset (in batch iteration order)
        dataset_idx_offset = 0

        # for the later check: which classes had errors
        error_classes: set[int] = set()
        last_batch_size: int = dataloader.batch_size or 1

        with torch.no_grad():
            total_size = len(dataloader.dataset)
            progress = tqdm(
                initial=0,
                total=total_size,
                desc=f"--> Evaluating ({mode}) ",
                ncols=0,
            )

            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.net(images)

                # update the metric on the main dataset
                self.metric.update(logits, labels)

                # compute predictions
                if logits.ndim > 1 and logits.size(-1) > 1:
                    preds = torch.argmax(logits, dim=-1)
                else:
                    preds = (logits > 0).long().view_as(labels)

                wrong_mask = preds != labels
                if wrong_mask.any():
                    wrong_idx = torch.nonzero(wrong_mask, as_tuple=False).squeeze(1)

                    for wi in wrong_idx:
                        wi_int = int(wi.item())
                        img = images[wi_int].detach().cpu()
                        true_cls = int(labels[wi_int].detach().cpu())
                        pred_cls = int(preds[wi_int].detach().cpu())

                        # remember the class that had errors
                        error_classes.add(true_cls)

                        # folder by TRUE class: errors_root/<true_cls>
                        subdir = os.path.join(self.errors_root, f"{true_cls}")
                        os.makedirs(subdir, exist_ok=True)

                        # global index of the sample in the dataset (in traversal order)
                        dataset_global_idx = dataset_idx_offset + wi_int

                        saved = False
                        if has_source_path:
                            try:
                                src_path = dataloader.dataset.get_source_path(
                                    dataset_global_idx
                                )
                            except TypeError:
                                # in case of the old signature without idx
                                src_path = dataloader.dataset.get_source_path()

                            if src_path is not None and os.path.exists(src_path):
                                # save the source file itself
                                dst_name = os.path.basename(src_path)
                                dst_path = os.path.join(subdir, dst_name)
                                shutil.copy2(src_path, dst_path)
                                saved = True

                        # fallback: if the source file could not be obtained/copied,
                        # save the transformed image
                        if not saved:
                            fname = f"{global_idx:08d}.png"
                            fpath = os.path.join(subdir, fname)
                            save_image(img, fpath)
                            global_idx += 1

                batch_size = len(images)
                last_batch_size = batch_size
                dataset_idx_offset += batch_size
                progress.update(batch_size)

            progress.close()

        # main result on the original dataset
        result = self.metric.compute()
        if torch.is_tensor(result) and result.numel() == 1:
            result = float(result.cpu())

        print(f"--> Main metric on validation ({mode}): {result}")

        # additional check on the collected errors:
        # use the same augment/preprocess as in the original dataset
        self._evaluate_on_errors(
            batch_size=last_batch_size,
            error_classes=error_classes,
            base_dataset=dataloader.dataset,
        )

        return result