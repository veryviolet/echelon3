"""Synthetic YOLO-format detection dataset for the detector smoke.

Each image has 1-3 axis-aligned bright rectangles on a dark noisy background.
The class is decided by aspect ratio (0 = wide, 1 = tall), so the detector has a
learnable signal and mAP climbs above zero within a few dozen epochs.

Layout (standard YOLO):
    root/images/<split>/img_XXXX.png
    root/labels/<split>/img_XXXX.txt      # lines: "cls xc yc w h" (normalized)
"""
import argparse
import os
import numpy as np
import cv2


def gen_split(root, split, n, size, seed):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(root, 'images', split)
    lbl_dir = os.path.join(root, 'labels', split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(n):
        img = rng.integers(0, 40, size=(size, size, 3), dtype=np.uint8)
        lines = []
        for _ in range(int(rng.integers(1, 4))):
            cls = int(rng.integers(0, 2))
            if cls == 0:       # wide
                w = rng.uniform(0.25, 0.45); h = rng.uniform(0.10, 0.20)
            else:              # tall
                w = rng.uniform(0.10, 0.20); h = rng.uniform(0.25, 0.45)
            xc = rng.uniform(w / 2 + 0.02, 1 - w / 2 - 0.02)
            yc = rng.uniform(h / 2 + 0.02, 1 - h / 2 - 0.02)
            x1 = int((xc - w / 2) * size); y1 = int((yc - h / 2) * size)
            x2 = int((xc + w / 2) * size); y2 = int((yc + h / 2) * size)
            color = (220, 220, 220) if cls == 0 else (200, 230, 200)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        cv2.imwrite(os.path.join(img_dir, f'img_{i:04d}.png'), img)
        with open(os.path.join(lbl_dir, f'img_{i:04d}.txt'), 'w') as f:
            f.write("\n".join(lines) + "\n")

    print(f'{split}: {n} images -> {img_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./det_data')
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--train', type=int, default=256)
    ap.add_argument('--test', type=int, default=48)
    args = ap.parse_args()
    gen_split(args.root, 'train', args.train, args.size, seed=1)
    gen_split(args.root, 'test', args.test, args.size, seed=2)
