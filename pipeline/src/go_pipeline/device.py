"""Device resolution: a GPU when available (NVIDIA CUDA, or AMD ROCm
through torch's HIP build, which answers to the same `cuda` device
type), CPU otherwise.

The same code path runs on all three. On CUDA we enable TF32 matmuls, cuDNN
benchmarking (unless deterministic mode is requested), pinned-memory
loaders and automatic mixed precision.
"""
from __future__ import annotations

import torch


def is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(name)
    if is_cuda(device) and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install the CUDA torch wheel (see pipeline/pyproject.toml, which "
            "pins the PyTorch CUDA index on Linux x86_64) and check `nvidia-smi`, "
            "or the ROCm wheel on AMD (see pipeline/README.md)."
        )
    return device


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 on CUDA hardware that supports it, fp16 next, None on CPU."""
    if not is_cuda(device):
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def configure(device: torch.device, deterministic: bool = False) -> None:
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if is_cuda(device):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def describe(device: torch.device) -> str:
    if is_cuda(device):
        props = torch.cuda.get_device_properties(device)
        backend = f"rocm {torch.version.hip}, {getattr(props, 'gcnArchName', '?')}" if torch.version.hip \
            else f"capability {props.major}.{props.minor}"
        return f"cuda: {props.name} ({props.total_memory / 2**30:.1f} GiB, {backend})"
    return "cpu"


if __name__ == "__main__":
    _device = resolve_device("auto")
    configure(_device)
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | device: {describe(_device)}")
