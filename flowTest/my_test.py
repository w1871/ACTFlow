import os
import numpy as np
from util.frame_utils import readFlow
import torch

FLO = readFlow('frame_0002.flo')
print(type(FLO))
print(FLO.shape)
# img = Image.open('/home/dell/TapLab-master/1/frame_0002.jpg').convert('RGB')
# img = numpy.array(img)
# print(img)
# img = transforms.ToTensor()(img)
#
# print(img.shape)
# img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
# print(img)

# def find_max_value(tensor):
#     max_value = np.max(tensor)
#     return max_value
#
# # 示例张量
# tensor = np.random.rand(256, 1024, 2)
# print(tensor)
# max_value = find_max_value(tensor)
# print("张量中的最大值为:", max_value)
#
# flo = torch.Tensor(FLO)
# flo = flo.permute(2, 0, 1)
# print(flo)
# print(flo.shape)