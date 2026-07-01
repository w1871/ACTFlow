import torch
import torch.nn.functional as F

# aa = torch.randn(1,3,3,3)
# print(aa)
# b = F.interpolate(aa, (3, 3), mode='area')
# # print(b)
# a = 5
# a = a > 0.5
# print(a)
aa = torch.tensor([[[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],

                   [[1, 2, 3],
                    [4, 5, 6],
                    [1,2,3]]
                   ]],dtype=float)
weight,i = torch.topk(aa,k=2, dim=-1)
print(weight)
print(i)
# print(b.shape)
# print(aa.shape)
# aa = aa.mean([2, 3])
# print(aa)
# bb = torch.randn(1,3,256,1024)
# bb = bb.mean([2, 3])
# print(bb.shape)
# print(bb)
# cc = torch.randn(1,16,3)
# print(cc)
#
# bb = torch.tensor([[[[1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1]],
#
#                    [[1, 2, 3],
#                     [4, 5, 6],
#                     [7, 8, 9]]
#                    ]])
# print(aa.shape)
# print(aa.sum())
# cc = aa - bb
# print(cc.shape)
# print(cc)
# cc = cc.type(torch.DoubleTensor)
# print(cc)
# map = torch.norm(cc, p=2, dim=1)
# print(map.shape)
# print(map)