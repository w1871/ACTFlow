from PIL import Image
import numpy as np

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 20 * np.log10(255 / np.sqrt(mse))

a=0
for i in range(10,200,9):
    if i<100:
        #print('00'+str(i))
        img1 = np.array(Image.open('../checkpoints/no_atten/web/images/epoch0' + str(i) + '_fake_B.png')).astype(np.float64)
        img2 = np.array(Image.open('../checkpoints/no_atten/web/images/epoch0' + str(i) + '_real_B.png')).astype(np.float64)
        if __name__ == "__main__":
         p=psnr(img1, img2)
         print(str(i)+'-'+str(psnr(img1, img2)))
         a=a+p

# img1 = np.array(Image.open('../checkpoints/temple_3_U1/web/images/epoch100_fake_B.png')).astype(np.float64)
# img2 = np.array(Image.open('../checkpoints/temple_3_U1/web/images/epoch100_real_B.png')).astype(np.float64)
# if __name__ == "__main__":
#  p = psnr(img1, img2)
# print(100,p)
# a = a + p

for i in range(110, 201, 9):
        #print(str(i))
        img1 = np.array(Image.open('../checkpoints/temple_3_U1/web/images/epoch' + str(i) + '_fake_B.png')).astype(np.float64)
        img2 = np.array(Image.open('../checkpoints/temple_3_U1/web/images/epoch' + str(i) + '_real_B.png')).astype(np.float64)
        if __name__ == "__main__":
         p = psnr(img1, img2)
         print(str(i)+'-'+str(psnr(img1, img2)))
         a = a + p

print(a)





#img1 = np.array(Image.open('D:\RESONLY\images\epoch'+str(i)+'_fake_B.png')).astype(np.float64)
#img2 = np.array(Image.open('D:\RESONLY\images\epoch191_real_B.png')).astype(np.float64)
#img2 = np.array(Image.open('D:\RESONLY\images\epoch'+str(i)+'_real_B.png')).astype(np.float64)
#if __name__ == "__main__":
#   print(psnr(img1, img2))
