import torch
from bind_example.bind_matmul import mod

torch.backends.cuda.matmul.allow_tf32 = False

a = torch.rand(512, 256, device = "cuda")
b = torch.rand(256, 128, device = "cuda")
c = mod.matmul(a, b)
torch.testing.assert_close(c, a @ b, rtol = 1e-4, atol = 1e-3)
print("matched torch")

