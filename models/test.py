import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim=3, heads=3, dim_heads=None):
        super().__init__()
        print("参数",dim,heads,dim_heads)
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        print("SelfAttention", x.shape)

        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))
        print("qqq", q.shape)
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

image = torch.randn(1024,256,3)

# 定义 SelfAttention 模块
self_attention = SelfAttention(dim=3, heads=3)

# 将展平后的图像输入到 SelfAttention 中
output = self_attention(image)

print("Output shape:", output.shape)  # 查看输出的形状