import cv2
import numpy as np
import os
from PIL import Image
import numpy as np
def compute_epe(flow_gt, flow_pred):
    epe = np.sqrt(np.sum((img1.astype(float) - img2.astype(float)) ** 2))
    return epe


a=0
for i in range (10,200,9):
    if i<100:
        img1 = cv2.imread('../checkpoints/temple_3_U1/web/images/epoch0' + str(i) + '_fake_B.png',cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('../checkpoints/temple_3_U1/web/images/epoch0' + str(i) + '_real_B.png',cv2.IMREAD_GRAYSCALE)
        e=compute_epe(img1, img2)
        print(i,e)
        a+=e

img1 = cv2.imread('../checkpoints/temple_3_U1/web/images/epoch100_fake_B.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../checkpoints/temple_3_U1/web/images/epoch100_real_B.png', cv2.IMREAD_GRAYSCALE)
e = compute_epe(img1, img2)
print(100,e)
a+=e

for i in range(110, 201, 9):
        img1 = cv2.imread('../checkpoints/temple_3_U1/web/images/epoch' + str(i) + '_fake_B.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('../checkpoints/temple_3_U1/web/images/epoch' + str(i) + '_real_B.png', cv2.IMREAD_GRAYSCALE)
        e = compute_epe(img1, img2)
        print(i,e)
        a+=e
print(a)


