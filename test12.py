import torch
from axial_attention import AxialAttention


img = torch.randn(1, 3, 256, 1024)
attn = AxialAttention(
    dim = 3,             # embedding dimension
    dim_index = 1,       # where is the embedding dimension
    dim_heads = 32,      # dimension of each head. defaults to dim // heads if not supplied
    heads = 1,           # number of heads for multi-head attention
    num_dimensions = 2,  # number of axial dimensions (images is 2, video is 3, or more)
)
img_attn = attn(img)
print(img_attn.size())