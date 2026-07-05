"""Generate a tiny synthetic 2-class image dataset for the smoke config.

Class 0: dark noise images; class 1: bright noise images — trivially separable,
so one epoch of training must push accuracy well above 0.5.

Usage: python gen_smoke_data.py --root ./smoke_data
"""
import argparse
import os

import cv2
import numpy as np


def generate(root: str, per_class: int, size: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    for split, n in (('train', per_class), ('test', max(8, per_class // 4))):
        for cls in (0, 1):
            folder = os.path.join(root, split, str(cls))
            os.makedirs(folder, exist_ok=True)
            for i in range(n):
                # class 0: vertical stripes, class 1: horizontal stripes (+noise);
                # структурный признак, а не яркостный — так train/eval статистики
                # BatchNorm совпадают и смоук не зависит от running stats
                period = int(rng.integers(4, 9))
                phase = int(rng.integers(0, period))
                coords = np.arange(size)
                stripes = (((coords + phase) // period) % 2 * 200 + 25).astype(np.uint8)
                img = np.tile(stripes[np.newaxis, :] if cls == 0 else stripes[:, np.newaxis],
                              (size, 1) if cls == 0 else (1, size))
                img = np.stack([img] * 3, axis=-1).astype(np.int16)
                img += rng.integers(-20, 20, size=img.shape, dtype=np.int16)
                img = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(folder, f'img_{i:04d}.png'), img)
    print(f'smoke dataset written to {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./smoke_data')
    parser.add_argument('--per-class', type=int, default=128)
    args = parser.parse_args()
    generate(args.root, args.per_class)
