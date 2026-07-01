# 导入必要的库
import numpy as np

# 读取.flo光流文件并转换为张量
from util.frame_utils import readFlow




# 归一化光流张量
def normalize_flow(flow):
    flow_norm = np.linalg.norm(flow, axis=2, keepdims=True)
    flow_norm[flow_norm == 0] = 1.0
    normalized_flow = flow / flow_norm
    return normalized_flow

# 反归一化光流张量
def denormalize_flow(normalized_flow, magnitude):
    denormalized_flow = normalized_flow * magnitude
    return denormalized_flow

# 读取.flo文件
file_path = "frame_0002.flo"
# flow = read_flow_file(file_path)
flow = readFlow(file_path)
# 归一化光流张量
normalized_flow = normalize_flow(flow)

# 假设在某处有一个magnitude（模）值
magnitude = np.linalg.norm(flow, axis=2)

# 反归一化光流张量
denormalized_flow = denormalize_flow(normalized_flow, magnitude)

# 输出结果示例
print("原始光流张量：")
print(flow)
print("\n归一化后的光流张量：")
print(normalized_flow)
print("\n反归一化后的光流张量：")
print(denormalized_flow)
