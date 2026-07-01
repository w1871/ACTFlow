import os
import subprocess
import numpy as np
import cv2
from tqdm import tqdm

# ====================== 固定编码参数 ======================
X264_CFG = {
    "profile": "main",
    "preset": "fast",
    "bframes": 0,          # disable B-frames
    "keyint": 10,         # intra-frame period = 10
    "partitions": "all",  # all inter-partition modes enabled
}

QP_LIST = [22, 27, 32, 37]

def encode_h264_x264(raw_video_path, out_root="compressed_h264"):
    """
    输入原始YUV/MP4
    """
    os.makedirs(out_root, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(raw_video_path))[0]

    for qp in QP_LIST:
        out_mp4 = os.path.join(out_root, f"{base_name}_qp{qp}.mp4")
        cmd = [
            "ffmpeg",
            "-i", raw_video_path,
            "-c:v", "libx264",
            "-profile:v", X264_CFG["profile"],
            "-preset", X264_CFG["preset"],
            "-x264-params",
            f"bframes={X264_CFG['bframes']}:keyint={X264_CFG['keyint']}:partitions={X264_CFG['partitions']}:qp={qp}",
            "-an", "-y", out_mp4
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Encoded H.264 QP={qp} -> {out_mp4}")
    return out_root

# 码流解析MV、残差并转为RGB
def bitstream_mv_res_to_rgb(mp4_path, dataset_out="compressed_dataset"):
    """
    流程：码流解析MV/RES → 空间对齐 → 归一化[0,1] → 色彩映射RGB图
    """
    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    mv_out_dir = os.path.join(dataset_out, "mv_rgb", base_name)
    res_out_dir = os.path.join(dataset_out, "res_rgb", base_name)
    os.makedirs(mv_out_dir, exist_ok=True)
    os.makedirs(res_out_dir, exist_ok=True)

    # 获取视频分辨率
    cap = cv2.VideoCapture(mp4_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    # ---------- 提取运动矢量MV 滤镜：showvectors 原生码流矢量 ----------
    mv_raw_pattern = os.path.join(dataset_out, "tmp_mv_%04d.png")
    cmd_mv = [
        "ffmpeg", "-i", mp4_path,
        "-filter:v", "showvectors=mode=motion:grid=1",
        "-pix_fmt", "gray", "-f", "image2", mv_raw_pattern, "-y"
    ]
    subprocess.run(cmd_mv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # ---------- 提取残差RES：码流预测差值残差图 ----------
    res_raw_pattern = os.path.join(dataset_out, "tmp_res_%04d.png")
    cmd_res = [
        "ffmpeg", "-i", mp4_path,
        "-filter:v", "extractplanes=y,normalize",
        "-pix_fmt", "gray", "-f", "image2", res_raw_pattern, "-y"
    ]
    subprocess.run(cmd_res, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_idx = 0
    while True:
        mv_gray_path = mv_raw_pattern % frame_idx
        res_gray_path = res_raw_pattern % frame_idx
        if not os.path.exists(mv_gray_path) or not os.path.exists(res_gray_path):
            break

        # 读取灰度图
        mv_gray = cv2.imread(mv_gray_path, 0).astype(np.float32)
        res_gray = cv2.imread(res_gray_path, 0).astype(np.float32)

        # 1. Spatial alignment 空间对齐（分辨率统一w×h，这里已匹配）
        mv_gray = cv2.resize(mv_gray, (w, h))
        res_gray = cv2.resize(res_gray, (w, h))

        # 2. Global normalization to [0, 1] 全局归一化
        def norm_01(arr):
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if arr_max - arr_min < 1e-8:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min)

        mv_norm = norm_01(mv_gray)
        res_norm = norm_01(res_gray)

        # 3. Color mapping to RGB
        mv_uint = (mv_norm * 255).astype(np.uint8)
        res_uint = (res_norm * 255).astype(np.uint8)
        mv_rgb = cv2.applyColorMap(mv_uint, cv2.COLORMAP_JET)
        res_rgb = cv2.applyColorMap(res_uint, cv2.COLORMAP_JET)

        # 保存RGB图像
        cv2.imwrite(os.path.join(mv_out_dir, f"frame_{frame_idx:04d}.png"), mv_rgb)
        cv2.imwrite(os.path.join(res_out_dir, f"frame_{frame_idx:04d}.png"), res_rgb)

        # 删除临时灰度文件
        os.remove(mv_gray_path)
        os.remove(res_gray_path)
        frame_idx += 1
    print(f"Dataset generated for {base_name}, total frames: {frame_idx}")

# ====================== 批量处理入口 ======================
if __name__ == "__main__":
    # 原始视频路径
    RAW_VIDEO = "bamboo_original.mp4"

    # 编码4种QP的H.264压缩视频
    compressed_root = encode_h264_x264(RAW_VIDEO)

    # 提取MV/RES并生成RGB数据集
    mp4_list = [os.path.join(compressed_root, f) for f in os.listdir(compressed_root) if f.endswith(".mp4")]
    for mp4 in tqdm(mp4_list, desc="Generating compressed-domain dataset"):
        bitstream_mv_res_to_rgb(mp4, dataset_out="compressed_domain_dataset")