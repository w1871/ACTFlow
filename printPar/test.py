import torch

# 加载模型
model = torch.load('latest_net_G.pth')

# 获取模型参数
model_parameters = model.state_dict()

# 打印模型参数
for name, param in model_parameters.items():
    print(name, param.size())