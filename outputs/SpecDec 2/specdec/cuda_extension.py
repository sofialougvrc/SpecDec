"""ctypes loader for the standalone CUDA acceptance-rejection kernel.

This intentionally avoids ``torch.utils.cpp_extension.load_inline``. Colab can
compile inline extensions cleanly and still fail at Python import time; a normal
NVCC-built shared object plus ctypes is a smaller moving target.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CudaExtensionError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _detect_arch() -> str:
    try:
        import torch

        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    except Exception:
        return "sm_75"


def build_acceptance_rejection_so(
    *,
    output_dir: str | os.PathLike[str] = "build/cuda",
    arch: str | None = None,
    force: bool = False,
) -> Path:
    """Compile the CUDA kernel to a shared object with NVCC."""

    root = _repo_root()
    source = root / "specdec" / "cuda" / "acceptance_rejection_kernel.cu"
    out_dir = root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = out_dir / "libspecdec_acceptance_rejection.so"
    if so_path.exists() and not force:
        return so_path

    arch = arch or _detect_arch()
    command = [
        "nvcc",
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-arch={arch}",
        str(source),
        "-o",
        str(so_path),
    ]
    try:
        subprocess.run(command, check=True, cwd=root)
    except FileNotFoundError as exc:
        raise CudaExtensionError("nvcc was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise CudaExtensionError(f"nvcc failed with exit code {exc.returncode}") from exc
    return so_path


@dataclass
class AcceptanceRejectionCuda:
    """Callable Python wrapper around ``run_acceptance_rejection``."""

    so_path: Path

    def __post_init__(self) -> None:
        self.lib = ctypes.CDLL(str(self.so_path))
        ptr = ctypes.c_void_p
        self.lib.run_acceptance_rejection.argtypes = [
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulonglong,
            ctypes.c_int,
        ]
        self.lib.run_acceptance_rejection.restype = ctypes.c_int
        self.lib.run_acceptance_rejection_sync.argtypes = (
            self.lib.run_acceptance_rejection.argtypes
        )
        self.lib.run_acceptance_rejection_sync.restype = ctypes.c_int

    @classmethod
    def build(
        cls,
        *,
        output_dir: str | os.PathLike[str] = "build/cuda",
        arch: str | None = None,
        force: bool = False,
    ) -> "AcceptanceRejectionCuda":
        return cls(build_acceptance_rejection_so(output_dir=output_dir, arch=arch, force=force))

    def __call__(
        self,
        target_probs: Any,
        draft_probs: Any,
        draft_tokens: Any,
        accept_probs: Any,
        accepted: Any,
        corrected_probs: Any,
        *,
        seed: int = 0,
        threads_per_block: int = 128,
        sync: bool = False,
    ) -> None:
        self._validate_tensors(
            target_probs,
            draft_probs,
            draft_tokens,
            accept_probs,
            accepted,
            corrected_probs,
        )
        depth, vocab_size = target_probs.shape
        fn = self.lib.run_acceptance_rejection_sync if sync else self.lib.run_acceptance_rejection
        status = fn(
            ctypes.c_void_p(target_probs.data_ptr()),
            ctypes.c_void_p(draft_probs.data_ptr()),
            ctypes.c_void_p(draft_tokens.data_ptr()),
            ctypes.c_void_p(accept_probs.data_ptr()),
            ctypes.c_void_p(accepted.data_ptr()),
            ctypes.c_void_p(corrected_probs.data_ptr()),
            ctypes.c_int(vocab_size),
            ctypes.c_int(depth),
            ctypes.c_ulonglong(seed),
            ctypes.c_int(threads_per_block),
        )
        if status != 0:
            raise CudaExtensionError(f"CUDA kernel launch failed with cudaError_t={status}")

    @staticmethod
    def _validate_tensors(
        target_probs: Any,
        draft_probs: Any,
        draft_tokens: Any,
        accept_probs: Any,
        accepted: Any,
        corrected_probs: Any,
    ) -> None:
        import torch

        tensors = [target_probs, draft_probs, draft_tokens, accept_probs, accepted, corrected_probs]
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("all arguments must be torch.Tensor instances")
        if not all(tensor.is_cuda for tensor in tensors):
            raise ValueError("all tensors must live on CUDA")
        if target_probs.dtype != torch.float32 or draft_probs.dtype != torch.float32:
            raise TypeError("target_probs and draft_probs must be torch.float32")
        if corrected_probs.dtype != torch.float32 or accept_probs.dtype != torch.float32:
            raise TypeError("corrected_probs and accept_probs must be torch.float32")
        if draft_tokens.dtype != torch.int32 or accepted.dtype != torch.int32:
            raise TypeError("draft_tokens and accepted must be torch.int32")
        if target_probs.ndim != 2:
            raise ValueError("target_probs must have shape [depth, vocab_size]")
        if draft_probs.shape != target_probs.shape or corrected_probs.shape != target_probs.shape:
            raise ValueError("draft_probs and corrected_probs must match target_probs shape")
        depth = target_probs.shape[0]
        if draft_tokens.shape != (depth,) or accept_probs.shape != (depth,) or accepted.shape != (depth,):
            raise ValueError("draft_tokens, accept_probs, and accepted must have shape [depth]")
        if not all(tensor.is_contiguous() for tensor in tensors):
            raise ValueError("all tensors must be contiguous")
