#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：simple_yolo 
@File    ：build_dataset.py
@Author  ：kg5s55
@Description: 
"""
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, img_path,
                 imgsz=640,
                 augment=True,
                 batch_size=16,
                 stride=32,
                 single_cls=False,
                 hyp=None):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.batch_size = batch_size
        self.stride = stride

        self.labels = self.get_labels()
        self.transforms = self.build_transforms(hyp=hyp)

    def build_transforms(self, hyp):
        pass

    def get_img_files(self, img_path):
        pass

    def get_labels(self, ):
        pass

    def get_image_and_label(self, index):
        pass

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(index))

    def __len__(self):
        return len(self.labels)
