## CUDA Kernels

This repository contains CUDA kernel implementations and test harnesses. Each kernel and harness is written from scratch and profiled using `ncu`. Assistance was utilized from LLM tools.

### Instructions

1. Setup CUDA

```
sudo apt update
sudo apt install -y build-essential
sudo apt install -y cuda-toolkit
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc
```

2. Compile the Kernel

```
nvcc vec_add.cu -o vec_add
```
If using Colab, then add stateless file writing to virtual disk.

```
%%writefile
#include <cstdio>
...
```

3. Run the Kernel

```
./vec_add
```

4. Profile using ncu

Profile for launch, sync and memory access errors.

```
compute-sanitizer ./vec_add
```

Profile timing and memory.

```
ncu -k regex:vec_add --launch-skip 1 --launch-count 1 ./vec_add
```

Thorough profiling against peak utilization and rooflines can be done via the GUI.

```
ncu --set full -k regex:vec_add --launch-skip 10 --launch-count 1 -o report ./vec_add
open -a "NVIDIA Nsight Compute" report.ncu-rep
```

5. Bind Kernel to Python

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"<KERNEL_WITH_TORCH_CPP_WRAPPER>"
cpp_src = r"torch::Tensor vec_add(torch::Tensor A, torch::Tensor B);"
mod = load_inline(
    name="my_vec_add",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["vec_add"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=True,
)

ans = mod.vec_add(torch.rand(2, device = "cuda"), torch.rand(2, device = "cuda")) # (2,)
```

