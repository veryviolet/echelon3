"""Synthetic semantic-segmentation dataset for the segmentation smoke.

Each image has a filled circle (class 1) and a filled rectangle (class 2) on a
dark noisy background (class 0). The mask stores the class index per pixel, so a
Segmenter has a clear learnable signal and mean-IoU climbs above zero.

Layout:
    root/images/<split>/img_XXXX.png     # RGB
    root/masks/<split>/img_XXXX.png       # single-channel class indices (0/1/2)
"""
import argparse
import os
import numpy as np
import cv2


def gen_split(root, split, n, size, seed):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(root, 'images', split)
    msk_dir = os.path.join(root, 'masks', split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for i in range(n):
        img = rng.integers(0, 40, size=(size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)

        # circle -> class 1
        r = int(rng.integers(size // 10, size // 5))
        cx = int(rng.integers(r, size - r)); cy = int(rng.integers(r, size - r))
        cv2.circle(img, (cx, cy), r, (230, 120, 120), thickness=-1)
        cv2.circle(mask, (cx, cy), r, 1, thickness=-1)

        # rectangle -> class 2
        w = int(rng.integers(size // 6, size // 3)); h = int(rng.integers(size // 6, size // 3))
        x1 = int(rng.integers(0, size - w)); y1 = int(rng.integers(0, size - h))
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (120, 120, 230), thickness=-1)
        cv2.rectangle(mask, (x1, y1), (x1 + w, y1 + h), 2, thickness=-1)

        cv2.imwrite(os.path.join(img_dir, f'img_{i:04d}.png'), img)
        cv2.imwrite(os.path.join(msk_dir, f'img_{i:04d}.png'), mask)

    print(f'{split}: {n} images -> {img_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./seg_data')
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--train', type=int, default=256)
    ap.add_argument('--test', type=int, default=48)
    args = ap.parse_args()
    gen_split(args.root, 'train', args.train, args.size, seed=1)
    gen_split(args.root, 'test', args.test, args.size, seed=2)
