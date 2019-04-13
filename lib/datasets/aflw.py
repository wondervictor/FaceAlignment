
import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import shufflelr, crop


class AFLW(data.Dataset):
    """AFLW

    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    """
    def __init__(self, cfg, image_set, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAIN_CSV
        else:
            self.csv_file = cfg.DATASET.TEST_CSV

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
        self.landmarks_frame = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]
        box_size = self.landmarks_frame.iloc[idx, 2]

        center_w = self.landmarks_frame.iloc[idx, 3]
        center_h = self.landmarks_frame.iloc[idx, 4]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 5:].values
        pts = pts.astype('float').reshape(-1, 2)
        pts = torch.Tensor(pts.tolist())

        scale *= 1.5
        nparts = pts.size(0)
        img = np.array(Image.open(image_path).convert('RGB'))

        # transform !
        # img = self.transform(img)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = shufflelr(pts, width=img.shape[1], dataset='aflw')
                center[0] = img.shape[1] - center[0]
                center_w = img.shape[1] - self.landmarks_frame.iloc[idx, 3]

        img = crop(img, center, scale, self.input_size, rot=r)
        img = self.transform(img)


if __name__ == '__main__':

    pass
