import cv2


def get_qp_value(frame):
    # 计算图像的平均亮度值，作为 QP 值的近似
    qp_value = frame.mean()
    return qp_value


def main():
    # 视频文件路径
    filename = 'alley_1_264_qp22.avi'

    # 打开视频文件
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("无法打开视频文件！")
        return

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            break

        # 获取当前帧的 QP 值
        qp_value = get_qp_value(frame)

        # 打印 QP 值
        print("当前视频帧的 QP 值：", qp_value)

        # 显示视频帧（可选）
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
