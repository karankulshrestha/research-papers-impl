import torch

temp = torch.randn(10, 8)

batch, size_k = temp.size()

print(batch, size_k)