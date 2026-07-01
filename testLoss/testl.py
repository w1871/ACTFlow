import torch

# 2 3 4
aa = torch.tensor([[[1, 2, 3,4],
                [1, 2, 3,4],
                [1, 2, 3,4]],

                [[-1, 2, 3,4],
                [1, 2, 3,4],
                [1, -2, 3,4]]]
                )
print(aa.shape)
aa = torch.Tensor(aa).permute(2, 0, 1).float()
# 4 2 3
print(aa.shape)
print(aa)
print("aa[0]",aa[0].shape)
print("aa[0]",aa[0])
print("aa[1]",aa[1].shape)
print("aa[1]",aa[1])
print("aa[0].abs()",aa[0].abs())
print("aa[1].abs()",aa[1].abs())
bb = (aa[0].abs() < 10) & (aa[1].abs() < 10)
print(bb.shape)
print(bb)

