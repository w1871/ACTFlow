from PIL import Image

import os
from PIL import Image

# 设置图片目录路径
image_directory = 'F:\.deeplearn\gan_flow\\aDeal\qp22\kitti\\train'  # 替换为实际的目录路径

# 支持的图片文件格式
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# 遍历目录下的所有文件
image_files = []
for filename in os.listdir(image_directory):
    # 获取文件的完整路径
    file_path = os.path.join(image_directory, filename)

    # 检查文件是否是图片格式
    if filename.lower().endswith(valid_extensions):
        # 尝试打开图片文件
        try:
            image = Image.open(file_path)
            width, height = image.size
            # 裁剪区域：从宽度2048开始，长度保持3072不变
            crop_box = (2048, 0, width, height)
            cropped_image = image.crop(crop_box)
            str = 'F:\.deeplearn\gan_flow\\aDeal\k2015_flow_img\\' + filename
            # 保存裁剪后的图片
            cropped_image.save(str)
            image_files.append(image)
            print(f"图片 {filename} 成功加载")
        except Exception as e:
            print(f"无法加载图片 {filename}: {e}")

# 输出加载的图片数
print(f"共加载了 {len(image_files)} 张图片")


