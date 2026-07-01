import os

import numpy as np
import torch
from util.frame_utils import readFlow
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_flo
from data.image_folder import make_dataset
from PIL import Image
#aligned_dataset.py包含一个可以加载图像对的数据集类。它设置好了一个图像目录/path/to/data/train，
#其中包含 {A,B} 形式的图像对。在测试期间，您需要准备一个目录/path/to/data/test作为测试数据。

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During testLoss time, you need to prepare a directory '/path/to/data/testLoss'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)  # get the image directory       获取数据路径
        self.dir_flo = os.path.join(opt.floroot, opt.phase)  # get the flo root 获取flo路径
        print("路径",self.dir_flo)
        print("路径", self.dir_ABC)
        self.ABC_paths = sorted(make_dataset(self.dir_ABC, opt.max_dataset_size,a=1))  # get image paths  返回图像列表
        self.flo_paths = sorted(make_dataset(self.dir_flo, opt.max_dataset_size,a=2))
        print("路径", self.ABC_paths)
        print("路径",self.flo_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image  确保裁剪大小小于图片本身大小
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        ABC_path = self.ABC_paths[index]
        flo_path = self.flo_paths[index]
        ############################
        # def load_flow_to_numpy(path):
        #     with open(path, 'rb') as f:
        #         magic = np.fromfile(f, np.float32, count=1)
        #         assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        #         h = np.fromfile(f, np.int32, count=1)[0]
        #         w = np.fromfile(f, np.int32, count=1)[0]
        #         data = np.fromfile(f, np.float32, count=2 * w * h)
        #     data2D = np.resize(data, (w, h, 2))
        #     FLO = data2D[180:, :, :]
        #     return FLO
        ############################
        FLO = readFlow(flo_path)

        ABC = Image.open(ABC_path).convert('RGB')          #单独获取一张图片并且转换为RGB
        # split AB image into A and B
        w, h = ABC.size        #获取宽和高
        w2 = int(w / 3)
        w3 = 2*w2
        A = ABC.crop((0, 0, w2, h))     #对齐两幢图片    从左至右依次是A,C,B
        C = ABC.crop((w2, 0, w3, h))
        B = ABC.crop((w3, 0, w, h))
       # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))


        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        # 1024 3 256
        # BB = torch.Tensor(B).permute(2, 0, 1).float()

        print("shapeC",C.shape)
        print("类型",type(FLO))
        print("大小", FLO.shape)

        FLO_transform = get_transform_flo(self.opt, transform_params, grayscale=(self.output_nc == 1))
        flo = FLO_transform(FLO)
        print("flo最终大小",flo.shape)

        BB = B
        valid1 = (BB[0].abs() < 1000) & (BB[1].abs() < 1000)
        valid = valid1.float()
        return {'A': A, 'B': B, 'C': C,'flo':flo,'valid':valid,'A_paths': ABC_path, 'B_paths': ABC_path,'C_paths': ABC_path,'flo_path':flo_path,}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
