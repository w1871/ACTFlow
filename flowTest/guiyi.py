import numpy as np


def spatial_transform(self, flow1, flow2):
    pad_t = 0
    pad_b = 0
    pad_l = 0
    pad_r = 0

    if pad_b != 0 or pad_r != 0 or pad_t != 0:

        flow1 = np.pad(flow1, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant',
                       constant_values=((0, 0), (0, 0), (0, 0)))
        flow2 = np.pad(flow2, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant',
                       constant_values=((0, 0), (0, 0), (0, 0)))

    # randomly sample scale


    min_scale = np.maximum(
        (self.crop_size[0] + 1) / float(ht),
        (self.crop_size[1] + 1) / float(wd))

    scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
    scale_x = np.clip(scale, min_scale, None)
    scale_y = np.clip(scale, min_scale, None)

    if np.random.rand() < self.spatial_aug_prob:
        # rescale the images
        img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        img3 = cv2.resize(img3, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        flow1, valid1 = self.resize_sparse_flow_map(flow1, valid1, fx=scale_x, fy=scale_y)
        flow2, valid2 = self.resize_sparse_flow_map(flow2, valid2, fx=scale_x, fy=scale_y)

    if self.do_flip:
        if np.random.rand() < 0.5:  # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            img3 = img3[:, ::-1]
            flow1 = flow1[:, ::-1] * [-1.0, 1.0]
            flow2 = flow2[:, ::-1] * [-1.0, 1.0]
            valid1 = valid1[:, ::-1]
            valid2 = valid2[:, ::-1]

    margin_y = 20
    margin_x = 50

    y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
    x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

    y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
    x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

    img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
    img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
    img3 = img3[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

    flow1 = flow1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
    flow2 = flow2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

    valid1 = valid1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
    valid2 = valid2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

    return img1, img2, img3, flow1, flow2, valid1, valid2