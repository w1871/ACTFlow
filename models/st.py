import torch
from torch import nn



# proj = nn.Conv2d(3, 48, kernel_size=3,stride=1, padding=1, bias=False)
# aa = torch.ones(1,3,256,1024)
# print(aa)
# bb = proj(aa)
# print(bb)

x = torch.randn(1,3,256,1024)
qkv = nn.Linear(3,9, bias=False)

q, kv = qkv(x).split([3, 6], dim=-1)
print(q.shape)
print(kv.shape)