# CenterNet-like detector smoke

A minimal, self-contained detection run: a `HeatmapDetector` (CenterNet-style
heatmap + width/height regression) trained on a synthetic YOLO-format dataset.
It exercises the whole detection path — YOLO dataset loading, variable-count box
collation, gaussian heatmap encoding, focal + width/height loss, peak decoding
and the `mAP` metric — with a tiny dependency-free backbone.

## Run

```bash
cd examples/detector

# 1. synthesize a YOLO dataset (images/<split>/*.png + labels/<split>/*.txt)
python gen_det_data.py --root ./det_data

# 2. train (needs the detection extra for the mAP COCO backend)
pip install "echelon3[detection]"
DET_DATA=./det_data echelon3-train --config-dir . --config-name detector_smoke

# CPU-only:
DET_DATA=./det_data echelon3-train --config-dir . --config-name detector_smoke device=cpu
```

`mAP` starts near zero and climbs past ~0.1 within ~40 epochs — the point is to
show the pipeline learns to localize, not to reach a strong number on toy data.

## Bring your own backbone

`HeatmapDetector` only requires a backbone that outputs `object_classes + 2`
channels at `output_size`. `backbone.py` is a four-layer toy; swap in any module
(e.g. a `Segmenter` with a `timm` or TopFormer backbone) by pointing
`net.config.heatmap_backbone` at it. The CLI puts the current directory on
`sys.path`, so `module: backbone` resolves to the file next to the config.
