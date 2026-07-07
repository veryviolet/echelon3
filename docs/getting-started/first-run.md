# First Run

This walkthrough trains a classifier end to end on a synthetic dataset that ships
with the repository, then exports it to ONNX. It exercises the whole pipeline in
under a minute on CPU.

## 1. Get the example files

```bash
git clone https://github.com/veryviolet/echelon3
cd echelon3
pip install -e ".[export]"
```

`examples/` contains:

- `gen_smoke_data.py` — writes a tiny 2-class image dataset (vertical vs.
  horizontal stripes) under `./smoke_data/{train,test}/{0,1}/`.
- `configs/smoke.yaml` — a complete config: a timm `mobilenetv3_small_100`
  backbone + a dense head, weighted cross-entropy, accuracy, AdamW, and an ONNX
  export section.

## 2. Generate data and train

```bash
python examples/gen_smoke_data.py --root ./smoke_data

SMOKE_DATA_ROOT=./smoke_data SMOKE_TARGET=./targets/smoke \
    echelon3-train --config-dir examples/configs --config-name smoke
```

`--config-dir` tells Hydra where your configs live; `--config-name` selects one
(without the `.yaml`). The two environment variables are read by the config via
`${oc.env:...}` defaults — the config itself carries no absolute paths.

You will see the framework build each component in turn (augmentations →
preprocess → datasets → dataloaders → network → losses → metrics → optimizer →
scheduler → checkpoint manager → logger → trainer), then train. The dataset is
trivially separable, so accuracy reaches 1.0 within a few epochs. Checkpoints and
TensorBoard event files are written under `SMOKE_TARGET`.

!!! tip "Override anything on the command line"
    Every config value is a Hydra override. To train longer:

    ```bash
    echelon3-train --config-dir examples/configs --config-name smoke \
        trainer.config.epochs=10 optimizer.config.lr=0.0005
    ```

## 3. Export to ONNX

The smoke config includes an `export` section, so:

```bash
SMOKE_TARGET=./targets/smoke \
    echelon3-export --config-dir examples/configs --config-name smoke
```

writes `./targets/smoke/smoke.onnx`, wrapping the preprocessing and the network
into a single graph that accepts a raw `uint8` NCHW image. Verify it with
onnxruntime, or read [Exporting to ONNX](../guide/export.md) for the details.

## Where to go next

- [How It Works](../concepts/how-it-works.md) — the `module`/`type`/`config` model.
- [Config Schema](../reference/config-schema.md) — every section of a config.
- [Extending](../guide/extending.md) — plug in your own networks and datasets.
