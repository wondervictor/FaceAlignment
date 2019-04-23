
import os
import math
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
from scipy.io import loadmat

from ..utils.transforms import shufflelr, crop, get_labelmap, transform_pixel


class COFW(data.Dataset):
    """COFW

    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    """
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.mat_file = cfg.DATASET.TRAIN_ANNOTATION
        else:
            self.mat_file = cfg.DATASET.TEST_ANNOTATION

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.mat = loadmat(self.mat_file)
        if is_train:
            self.images = self.mat['IsTr']
            self.pts = self.mat['phisTr']
        else:
            self.images = self.mat['IsT']
            self.pts = self.mat['phisT']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx][0]

        # grayscale --> RGB
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)

        pts = self.pts[idx][0:58].reshape(2, -1).transpose()

        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])

        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        center = torch.Tensor([center_w, center_h])

        # pts = torch.Tensor(pts.tolist())

        scale *= 1.25
        nparts = pts.shape[0]

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0

            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = shufflelr(pts, width=img.shape[1], dataset='cofw')
                center[0] = img.shape[1] - center[0]
                # center_w = img.shape[1] - self.landmarks_frame.iloc[idx, 3]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.clone()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                target[i] = get_labelmap(target[i], tpts[i]-1, self.sigma,
                                         label_type=self.label_type)

        img = self.transform(img)
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta


if __name__ == '__main__':

    pass
