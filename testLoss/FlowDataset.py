import data
import numpy as np
import torch
import torch.utils.data as data


import random

from testLoss import frame_utils
from testLoss.augmentor import FlowAugmentor, SparseFlowAugmentor



import copy

class FlowData(data.base_dataset):
    def __init__(self, aug_params=None, sparse=False, oneside=False, reverse_rate=0.3,realA=None,realB=None,realC=None):
        self.augmentor = None
        self.sparse = sparse
        self.oneside = oneside
        self.reverse_rate = reverse_rate
        print("[reverse_rate is {}]".format(self.reverse_rate))

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.realA = realA
        self.realB = realB
        self.realC = realC

    def __getitem__(self, index):
        # print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img3 = frame_utils.read_gen(self.image_list[index][2])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img3 = np.array(img3).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            img3 = torch.from_numpy(img3).permute(2, 0, 1).float()

            return torch.stack([img1, img2, img3]), self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True


        valid1 = valid2 = None

        if self.oneside:
            if self.sparse:
                flow1, valid1 = frame_utils.readFlowKITTI(self.flow_list[index])
                flow2 = copy.deepcopy(flow1)
                valid2 = copy.deepcopy(valid1) * 0
            else:
                flow1 = frame_utils.read_gen(self.flow_list[index])
                flow2 = copy.deepcopy(flow1) * 0 + 10000

        else:
            flow1 = frame_utils.read_gen(self.flow_list[index][0])
            flow2 = frame_utils.read_gen(self.flow_list[index][1])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img3 = frame_utils.read_gen(self.image_list[index][2])

        flow1 = np.array(flow1).astype(np.float32)
        flow2 = np.array(flow2).astype(np.float32)

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img3 = np.array(img3).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
            img3 = np.tile(img3[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            img3 = img3[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, img3, flow1, flow2, valid1, valid2 = self.augmentor(img1, img2, img3, flow1, flow2,
                                                                                valid1, valid2)
            else:
                img1, img2, img3, flow1, flow2 = self.augmentor(img1, img2, img3, flow1, flow2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img3 = torch.from_numpy(img3).permute(2, 0, 1).float()

        flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
        flow2 = torch.from_numpy(flow2).permute(2, 0, 1).float()

        if valid1 is not None and valid2 is not None:
            valid1 = torch.from_numpy(valid1)
            valid2 = torch.from_numpy(valid2) * 0  # sparse must be oneside
        else:
            valid1 = (flow1[0].abs() < 1000) & (flow1[1].abs() < 1000)
            valid2 = (flow2[0].abs() < 1000) & (flow2[1].abs() < 1000)

        if np.random.rand() < self.reverse_rate:
            return torch.stack([img3, img2, img1]), torch.stack([flow2, flow1]), torch.stack(
                [valid2.float(), valid1.float()])
        else:
            return torch.stack([img1, img2, img3]), torch.stack([flow1, flow2]), torch.stack(
                [valid1.float(), valid2.float()])

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)
