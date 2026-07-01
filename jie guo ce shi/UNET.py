from PIL import Image
import numpy as np

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 20 * np.log10(255 / np.sqrt(mse))

a=0
img1 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch037_fake_B.png')).astype(np.float64)
img2 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch037_real_B.png')).astype(np.float64)
if __name__ == "__main__":
 p = psnr(img1, img2)
 print(str(psnr(img1, img2)))
 a+=p

 img1 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch073_fake_B.png')).astype(np.float64)
 img2 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch073_real_B.png')).astype(np.float64)
 if __name__ == "__main__":
     p = psnr(img1, img2)
     print(str(psnr(img1, img2)))
 a+=p

img1 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch110_fake_B.png')).astype(np.float64)
img2 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch110_real_B.png')).astype(np.float64)
if __name__ == "__main__":
 p = psnr(img1, img2)
 print(str(psnr(img1, img2)))
 a+=p

 img1 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch146_fake_B.png')).astype(np.float64)
 img2 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch146_real_B.png')).astype(np.float64)
 if __name__ == "__main__":
     p = psnr(img1, img2)
     print(str(psnr(img1, img2)))
 a+=p

img1 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch182_fake_B.png')).astype(np.float64)
img2 = np.array(Image.open('../checkpoints/ambush_2_U2/web/images/epoch182_real_B.png')).astype(np.float64)
if __name__ == "__main__":
 p = psnr(img1, img2)
 print(str(psnr(img1, img2)))
 a+=p


print(a)