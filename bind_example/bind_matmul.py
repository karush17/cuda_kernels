import os
import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path

# compile only for current device
major, minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

cuda_src = Path("./cuda_src.cpp").read_text(encoding = "utf-8")
cpp_src = "torch::Tensor matmul(torch::Tensor A, torch::Tensor B);"
mod = load_inline(
    name = "my_matmul",
    cpp_sources = cpp_src,
    cuda_sources = cuda_src,
    functions = ["matmul"],
    extra_cuda_cflags = ["-O3", "-lineinfo"],
    verbose = True
)