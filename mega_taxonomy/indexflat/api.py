from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def _build_extension() -> Any:
    root = Path(__file__).resolve().parents[2]
    build_dir = root / "build-cpp-omp"
    build_dir.mkdir(parents=True, exist_ok=True)
    torch_build_dir = build_dir / "torch_ext"
    torch_build_dir.mkdir(parents=True, exist_ok=True)

    extra_cflags = ["-O3", "-std=c++17", f"-I{root / 'include'}"]
    extra_ldflags: list[str] = [f"-L{build_dir}", "-lindexflat_c", f"-Wl,-rpath,{build_dir}"]
    if os.uname().sysname == "Darwin":
        extra_ldflags.extend(
            ["-Wl,-rpath,/opt/homebrew/opt/libomp/lib", "-L/opt/homebrew/opt/libomp/lib"]
        )

    cpp_ext = __import__("torch.utils.cpp_extension", fromlist=["load"])
    load_fn: Any = cpp_ext.load
    ext = load_fn(
        name="_indexflat_torch_ext",
        sources=[str(root / "mega_taxonomy" / "indexflat" / "torch_extension.cpp")],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        build_directory=str(torch_build_dir),
        verbose=False,
    )
    if ext is None:
        raise RuntimeError("failed to build/load torch extension")
    return ext


_EXT: Any = _build_extension()

DTYPE_F16 = int(_EXT.DTYPE_F16)
DTYPE_BF16 = int(_EXT.DTYPE_BF16)
DTYPE_F32 = int(_EXT.DTYPE_F32)
BACKEND_AUTO = int(_EXT.BACKEND_AUTO)
BACKEND_CPU = int(_EXT.BACKEND_CPU)
BACKEND_CUDA = int(_EXT.BACKEND_CUDA)
BACKEND_METAL = int(_EXT.BACKEND_METAL)


class IndexFlatL2:
    def __init__(self, dim: int, dtype: int = DTYPE_F32) -> None:
        self._impl: Any = _EXT.IndexFlatL2(dim, dtype)

    @property
    def dim(self) -> int:
        return int(self._impl.dim)

    @property
    def ntotal(self) -> int:
        return int(self._impl.ntotal)

    def add(self, x: torch.Tensor) -> None:
        self._impl.add(x)

    def search(
        self,
        q: torch.Tensor,
        k: int,
        *,
        backend: int = BACKEND_AUTO,
        block_n: int = 256,
        block_d: int = 64,
        num_threads: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._impl.search(
            q,
            k,
            backend=backend,
            block_n=block_n,
            block_d=block_d,
            num_threads=num_threads,
        )
        return out[0], out[1]
