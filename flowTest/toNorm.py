import numpy as np
import torch

from util.frame_utils import readFlow

def find_max_value(tensor):
    max_value = np.max(tensor)
    return max_value

def normalize_flow(flow):
    flow_norm = np.linalg.norm(flow, axis=2, keepdims=True)
    flow_norm[flow_norm == 0] = 1.0
    normalized_flow = flow / flow_norm
    return normalized_flow

def denormalize_flow(normalized_flow):
    flow_norm = np.linalg.norm(normalized_flow, axis=2, keepdims=True)
    denormalized_flow = normalized_flow * flow_norm
    return denormalized_flow


# 读取.flo文件
file_path = "frame_0002.flo"
# flow = read_flow_file(file_path)
aa = readFlow(file_path)
# print(type(flow))
# print(flow.shape)

 # aa = np.random.rand(256, 1024, 2)
print(type(aa))
print(aa.shape)
print(aa)

b = normalize_flow(aa)
print(type(b))
print(b.shape)
print(b)
print("最大值",find_max_value(b))

c = denormalize_flow(b)
print(type(c))
print(c.shape)
print(c)
print("最大值",find_max_value(c))