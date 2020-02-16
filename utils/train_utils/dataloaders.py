import os
import random
import time

import numpy as np
import pandas as pd
import rawpy
import torch
from torch.utils.data import Dataset, DataLoader

from utils.train_utils.data_transforms import get_train_transform, get_eval_transform
import matplotlib.pyplot as plt

def raw2np(raw, black_level):
    im = raw.raw_image_visible.astype(np.float16)
    im = np.maximum(im - black_level, 0) / (16383 - black_level)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    return im


def pack_raw(im):
    # pack Bayer image to 4 channels
    img_shape = im.shape
    h = img_shape[0]
    w = img_shape[1]

    out = np.concatenate((im[0:h:2, 0:w:2, :],
                          im[0:h:2, 1:w:2, :],
                          im[1:h:2, 1:w:2, :],
                          im[1:h:2, 0:w:2, :]), axis=2)
    return out


class SIDDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, black_level=512, stack_bayer=False, subset=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = pd.read_csv(csv_file, delimiter=' ', header=None)
        self.images_frame = self.images_frame[np.random.rand(len(self.images_frame)) <= subset]
        self.root_dir = root_dir
        self.transform = transform
        self.im_dict = dict()
        self.gt_dict = dict()
        self.black_level = black_level
        self.stack_bayer = stack_bayer

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx, timing=False):
        t = {}
        t[0] = time.time()

        im_path = self.root_dir + self.images_frame.iloc[idx, 0][1:]
        gt_path = self.root_dir + self.images_frame.iloc[idx, 1][1:]
        _, im_fn = os.path.split(im_path)
        _, gt_fn = os.path.split(gt_path)
        im_exposure = float(im_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / im_exposure, 300)

        t[1] = time.time()
        if im_path in self.im_dict.keys():
            im = self.im_dict[im_path] * ratio
            # print('in image was restored',os.path.split(im_path)[-1])
        else:
            raw = rawpy.imread(im_path)
            # print('in image was read',os.path.split(im_path)[-1])
            im = raw2np(raw, self.black_level)
            if self.stack_bayer:
                im = pack_raw(im)
            self.im_dict[im_path] = im
            im *= ratio
        t[2] = time.time()
        im = im.astype(np.float32)
        t[3] = time.time()

        if gt_path in self.gt_dict.keys():
            gt = self.gt_dict[gt_path]
            # print('gt image was restored',os.path.split(gt_path)[-1])
        else:
            gt_raw = rawpy.imread(gt_path)
            # print('gt image was read',os.path.split(gt_path)[-1])
            gt = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt = np.float16(gt / 65535.0)
            gt = gt.astype(np.float32)
            self.gt_dict[gt_path] = gt
        t[4] = time.time()

        t[5] = time.time()

        sample = {'image': im, 'ground_truth': gt}
        t[6] = time.time()
        if self.transform:
            random.seed(0)
            np.random.seed(0)
            torch.random.manual_seed(0)
            sample = self.transform(sample)
        t[7] = time.time()

        if timing:
            tv = list(t.values())
            tk = list(t.keys())
            dt = [x-y for x, y in zip(tv[1:], tv[:-1])]
            dt_ticks = [f'{x}-{y}' for x, y in zip(tk[1:], tk[:-1])]
            plt.title(f'total time: {round(tv[-1]-tv[0],4)}')
            plt.xticks(tk[1:], dt_ticks)
            plt.plot(tk[1:], dt)
            for a, b in zip(tk[1:], dt):
                plt.text(a, b, str(round(b, 2)))
            plt.show()

        return sample['image'], sample['ground_truth']


class DataloadersComponent:

    def __init__(self, ingredient):
        @ingredient.capture(prefix="dataloader_cfg")
        def get_dataloaders(train_csv_path, eval_csv_path, root_dir, black_level, stack_bayer, subset, crop_size,
                            batch_size,
                            shuffle, num_workers, pin_memory):
            train_transform = get_train_transform(crop_size=crop_size, stack_bayer=stack_bayer)
            eval_transform = get_eval_transform(crop_size=crop_size, stack_bayer=stack_bayer)
            train_dataset = SIDDataset(train_csv_path, root_dir, transform=train_transform, black_level=black_level,
                                       stack_bayer=stack_bayer, subset=subset)
            eval_dataset = SIDDataset(eval_csv_path, root_dir, transform=eval_transform, black_level=black_level,
                                      stack_bayer=stack_bayer, subset=subset)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers, pin_memory=pin_memory)
            eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, pin_memory=pin_memory)

            return train_dataloader, eval_dataloader

        self.methods = [get_dataloaders]
