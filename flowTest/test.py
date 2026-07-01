import numpy as np

def find_max_value_in_flo(file_path):
    # 读取.flo文件
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError("不是合法的.flo文件格式")
        else:
            h = np.fromfile(f, np.int32, count=1)
            w = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
            data = np.resize(data, (h[0], w[0], 2))
            tensor = data

    max_value = np.min(tensor)
    return max_value

# 示例文件路径
file_path = "frame_0013.flo"
max_value = find_max_value_in_flo(file_path)
print("张量中的最大值为:", max_value)
