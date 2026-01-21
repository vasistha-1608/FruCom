import torch
import my_cuda_kernel 

x = torch.ones(100, device='cuda')
y = torch.ones(100, device='cuda') * 2
out = torch.zeros(100, device='cuda')


my_cuda_kernel.add(x, y, out)

if out[0] == 3.0:
    print("SUCCESS: Kernel returned 3.0")
else:
    print(f"FAILURE: Kernel returned {out[0]}")