import cv2
import imutils
from skimage.metrics import structural_similarity
import time
from PIL import Image
import numpy as np

start = time.time()
a=0
for i in range(1, 201):
    if i<10:
        src = cv2.imread('~/1pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/checkpoints/Fmvre_re_pix2pix/web/images/epoch00' + str(i) + '_fake_B.png')
        img = cv2.imread('~/1pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/checkpoints/Fmvre_re_pix2pix/web/images/epoch00' + str(i) + '_fake_B.png')
        grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算两个灰度图像之间的结构相似度
        (score, diff) = structural_similarity(grayA, grayB, win_size=101, full=True)
        diff = (diff * 255).astype("uint8")
        cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
        cv2.imshow("diff", diff)
        print(str(i)+'-' "SSIM:{}".format(score))
        s=score
        a=a+s
    elif i<100:
        src = cv2.imread('~/1pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/checkpoints/Fmvre_re_pix2pix/web/images/epoch0' + str(i) + '_fake_B.png')
        img = cv2.imread('~/1pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/checkpoints/Fmvre_re_pix2pix/web/images/epoch0' + str(i) + '_fake_B.png')
        grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算两个灰度图像之间的结构相似度
        (score, diff) = structural_similarity(grayA, grayB, win_size=101, full=True)
        diff = (diff * 255).astype("uint8")
        cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
        cv2.imshow("diff", diff)
        print(str(i)+'-' "SSIM:{}".format(score))
        s=score
        a=a+s
    else:
        src = cv2.imread('~/1pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/checkpoints/Fmvre_re_pix2pix/web/images/epoch' + str(i) + '_fake_B.png')
        img = cv2.imread('~/1pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/checkpoints/Fmvre_re_pix2pix/web/images/epoch' + str(i) + '_fake_B.png')
        grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算两个灰度图像之间的结构相似度
        (score, diff) = structural_similarity(grayA, grayB, win_size=101, full=True)
        diff = (diff * 255).astype("uint8")
        cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
        cv2.imshow("diff", diff)
        print(str(i)+'-' "SSIM:{}".format(score))
        s=score
        a=a+s
print(a)