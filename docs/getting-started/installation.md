# Installation

echelon3 requires Python 3.10+ and PyTorch 2.1+.

```bash
pip install echelon3
```

This pulls in the core dependencies (torch, torchvision, timm, torchmetrics,
albumentations, hydra-core, opencv-python, tensorboard, and friends) and installs
five console scripts on your `PATH`:

| Command | Purpose |
| --- | --- |
| `echelon3-train` | Train a model from a config. |
| `echelon3-evaluate` | Evaluate a checkpoint against a metric. |
| `echelon3-run` | Run inference over images/video with a runner. |
| `echelon3-export` | Export a checkpoint to ONNX. |
| `echelon3-finetune` | Train with warm-start, layer freezing and per-layer LR. |

## Optional extras

Some functionality lives behind extras so the base install stays lean:

```bash
pip install "echelon3[export]"   # ONNX export (onnx, onnxruntime)
pip install "echelon3[sam]"      # SAMOptimizer (mosaicml)
pip install "echelon3[smp]"      # segmentation-models-pytorch losses/necks
```

The corresponding modules import lazily: importing `echelon3` never fails because
an extra is missing — you only hit an `ImportError` (with a hint) if you actually
use a component that needs it.

!!! note "GPU vs CPU"
    echelon3 trains on CPU and GPU with no config changes. `pip install echelon3`
    pulls the default (CUDA) PyTorch wheels. For a CPU-only box, install the CPU
    build of torch first:

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install echelon3
    ```

## From source

```bash
git clone https://github.com/veryviolet/echelon3
cd echelon3
pip install -e ".[export]"
```

The repository ships an `examples/` directory with a synthetic-data generator and a
minimal classifier config used throughout this documentation and in CI.
