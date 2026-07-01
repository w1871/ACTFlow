import cv2
import numpy as np
import os
from PIL import Image
import numpy as np
def compute_epe(img1, img2):
    epe = np.sqrt(np.sum((img1.astype(float) - img2.astype(float)) ** 2))
    return epe

a=0
img1 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch037_fake_B.png')).astype(np.float64)
img2 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch037_real_B.png')).astype(np.float64)
e = compute_epe(img1, img2)
print(e)
a+=e


img1 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch073_fake_B.png')).astype(np.float64)
img2 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch073_real_B.png')).astype(np.float64)
e = compute_epe(img1, img2)
print(e)
a+=e

img1 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch110_fake_B.png')).astype(np.float64)
img2 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch110_real_B.png')).astype(np.float64)
e = compute_epe(img1, img2)
print(e)
a+=e

img1 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch146_fake_B.png')).astype(np.float64)
img2 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch146_real_B.png')).astype(np.float64)
e = compute_epe(img1, img2)
print(e)
a+=e

img1 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch182_fake_B.png')).astype(np.float64)
img2 = np.array(Image.open('../checkpoints/ambush_2_MvF_pix2pix/web/images/epoch182_real_B.png')).astype(np.float64)
e = compute_epe(img1, img2)
print(e)
a+=e

print(a)