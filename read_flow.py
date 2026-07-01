from PIL import  Image
import  os
import  numpy as np
aa="datasets/flow+/trainB/flow_x_00001.jpg"
img=Image.open (os.path.join(aa))
cc=np.array(img)
print((cc.shape))