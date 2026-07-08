"""Runtime performance knobs: mixed precision (AMP) and fast-matmul toggles.

Default precision is bf16 on GPUs that support it (Hopper/Ampere and newer) and
fp32 everywhere else — set `trainer.config.precision` to override
(`fp32` / `bf16` / `fp16` / `auto`). TF32 matmul and cuDNN autotuning are on by
default (`trainer.config.tf32`, `trainer.config.cudnn_benchmark`): large speedups
on modern GPUs at negligible risk for training.
"""
import torch


def resolve_amp_dtype(requested, device=None):
    """Map a precision string to an autocast dtype, or None for fp32.

    'auto' (the default) picks bf16 when the training device is a CUDA GPU that
    supports it, else fp32. `device` gates AMP to the actual device: on CPU (even
    on a box that also has a GPU) precision is always fp32.
    """
    req = str(requested).lower() if requested is not None else 'auto'
    is_cuda = torch.cuda.is_available()
    if device is not None:
        is_cuda = is_cuda and ('cuda' in str(getattr(device, 'type', device)))
    bf16_ok = is_cuda and torch.cuda.is_bf16_supported()

    if req in ('fp32', 'float32', '32', 'none', 'off', 'false'):
        return None
    if req in ('bf16', 'bfloat16'):
        return torch.bfloat16 if bf16_ok else None
    if req in ('fp16', 'float16', 'half', '16'):
        return torch.float16 if is_cuda else None
    if req == 'auto':
        return torch.bfloat16 if bf16_ok else None
    raise ValueError(f"unknown precision '{requested}' (expected auto/bf16/fp16/fp32)")


def precision_label(dtype):
    return {None: 'fp32', torch.bfloat16: 'bf16', torch.float16: 'fp16'}.get(dtype, str(dtype))


def setup_fast_matmul(tf32=True, cudnn_benchmark=True):
    """Enable TF32 matmul and cuDNN autotuning. No-op off CUDA."""
    if torch.cuda.is_available() and tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


def to_float32(x):
    """Recursively cast floating tensors in x to float32 (metrics after autocast).

    Handles a bare tensor, or the dict / list / tuple that detectors return.
    """
    if isinstance(x, torch.Tensor):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: to_float32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_float32(v) for v in x)
    return x
