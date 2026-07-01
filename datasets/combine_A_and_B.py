import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def image_write(path_A, path_C,path_B, path_ACB):        #A运动矢量 B光流 C残差
    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_C = cv2.imread(path_C, 1)
    im_ACB = np.concatenate([im_A,im_C,im_B], 1)
    cv2.imwrite(path_ACB, im_ACB)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../datasets/alley_2/mv_cont')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../datasets/alley_2/flow')
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image C', type=str, default='../datasets/alley_2/res_cont') #输入路径
parser.add_argument('--fold_ACB', dest='fold_ACB', help='output directory', type=str, default='../dataset/test_ACB')#输出目录
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_ACB', dest='use_ACB', help='if true: (0001_A, 0001_B, 0001_C) to (0001_ACB)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A)

if not args.no_multiprocessing:
    pool=Pool()

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_fold_C = os.path.join(args.fold_C, sp)
    img_list = os.listdir(img_fold_A)
    if args.use_ACB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_ACB = os.path.join(args.fold_ACB, sp)
    if not os.path.isdir(img_fold_ACB):
        os.makedirs(img_fold_ACB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_ACB:
            name_B = name_A.replace('_A.', '_B.')
            name_C = name_C.replace('_A.',  '_C.')
        else:
            name_B = name_A
            name_C = name_A
        path_B = os.path.join(img_fold_B, name_B)
        path_C = os.path.join(img_fold_C, name_C)
        if os.path.isfile(path_A) and os.path.isfile(path_B)and os.path.isfile(path_C):
            name_ACB = name_A
            if args.use_ACB:
                name_ACB = name_ACB.replace('_A.', '.')  # remove _A
            path_ACB = os.path.join(img_fold_ACB, name_ACB)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B,path_C,path_ACB))
            else:
                im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_C = cv2.imread(path_C, 1)
                im_ACB = np.concatenate([im_A, im_C, im_B], 1)
                cv2.imwrite(path_ACB, im_ACB)
if not args.no_multiprocessing:
    pool.close()
    pool.join()
