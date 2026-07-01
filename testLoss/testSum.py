import torch
# aa = torch.tensor([[[1, 2, 3,4],
#                 [1, 2, 3,4],
#                 [1, 2, 3,4]],
#
#                 [[-1, 2, 3,4],
#                 [1, 2, 3,4],
#                 [1, -2, 3,4]]]
#                 )
aa = torch.tensor([[[1, 2, 3,1],
                [4, 5, 6,1],
                [1, 1, 1,1]]]
                )
# 1 3 4
# dim = 1:按列 得出四个数的张量
# dim=2 ：按行，得出3个数的张量
print(aa.shape)
mag = torch.sum(aa,dim=2)
print(mag)

mag1 = torch.sum(aa,dim=-1)
print("mag1",mag1)