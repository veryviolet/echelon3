# Semantic-segmentation smoke

A minimal, self-contained segmentation run: a `Segmenter` (backbone → head →
bilinear upsample) trained on synthetic images whose per-pixel masks label a
circle (class 1) and a rectangle (class 2) over background (class 0). It exercises
the whole segmentation path — mask dataset loading, Albumentations mask
augmentation, cross-entropy loss and the `IoU` (mean-IoU) metric — with a tiny
dependency-free backbone.

## Run

```bash
cd examples/segmentation

# 1. synthesize an image/mask dataset (images/<split>/*.png + masks/<split>/*.png)
python gen_seg_data.py --root ./seg_data

# 2. train
SEG_DATA=./seg_data echelon3 train --config-dir . --config-name segmentation_smoke

# CPU-only:
SEG_DATA=./seg_data echelon3 train --config-dir . --config-name segmentation_smoke device=cpu
```

mean-IoU starts around 0.85 and climbs past ~0.95 within ~15 epochs — the point is
to show the pipeline learns to segment, not to reach a strong number on toy data.

## Bring your own backbone

`Segmenter` only needs a backbone that returns a single feature map for
`SimpleHead(channels, num_classes)`; it upsamples the head output back to the
input resolution itself. `backbone.py` is a four-layer toy — swap in any module
(e.g. a `timm`, DDRNet or TopFormer backbone) by pointing `net.config.backbone`
at it. The CLI puts the current directory on `sys.path`, so `module: backbone`
resolves to the file next to the config.
