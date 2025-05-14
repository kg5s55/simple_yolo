#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：simple_yolo 
@File    ：build_dataset.py
@Author  ：kg5s55
@Description: 
"""
import os
import glob
from pathlib import Path
import numpy as np

from itertools import repeat
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def verify_image_label(args):
    im_file, lb_file = args
    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found.
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"
                max_cls = lb[:, 0].max()  # max label count
                # assert max_cls <= num_cls, (
                #     f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                #     f"Possible class labels are 0-{num_cls - 1}"
                # )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    msg = f"WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        lb = lb[:, :5]
        return im_file, lb
    except Exception as e:
        # nc = 1
        msg = f"WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None]


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
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()

        self.transforms = self.build_transforms(hyp=hyp)

    def build_transforms(self, hyp):
        pass

    def get_labels(self, ):
        x = {"labels": []}
        self.label_files = img2label_paths(self.im_files)
        total = len(self.im_files)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,

                ),
            )
            # todo
            desc = "INFO"
            pbar = tqdm(results, desc=desc, total=total)
            for im_file, lb in pbar:
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            # "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
            pbar.close()


        #
        # try:
        #     # Verify images
        #     im = Image.open(im_file)

    def get_img_files(self, img_path):
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]
                else:
                    raise FileNotFoundError(f"{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert im_files, f"No images found in {img_path}. "
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {img_path}\n") from e

        return im_files

    def get_image_and_label(self, index):
        pass

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(index))

    def __len__(self):
        return len(self.labels)
