import torch
from torchvision import transforms
import numpy as np


def new_flip(x, dim):
    return torch.flip(x, [dim])


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class RandomFlips(object):

    def __call__(self, sample):
        im, gt = sample['image'], sample['ground_truth']

        if np.random.rand() > 0.5:
            im = new_flip(im, 1)
            gt = new_flip(gt, 1)
        if np.random.rand() > 0.5:
            im = new_flip(im, 2)
            gt = new_flip(gt, 2)
        return {'image': im, 'ground_truth': gt}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, stack_bayer):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.stack_bayer = stack_bayer

    def __call__(self, sample):
        im, gt = sample['image'], sample['ground_truth']

        h, w = im.shape[0:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        im = im[top: top + new_h, left: left + new_w, :]
        if self.stack_bayer:
            gt = gt[top * 2: top * 2 + new_h * 2, left * 2: left * 2 + new_w * 2, :]
        else:
            gt = gt[top: top + new_h, left: left + new_w, :]

        return {'image': im, 'ground_truth': gt}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['ground_truth']
        #         print(gt.shape)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = np.minimum(image, 1.0)
        gt = gt.transpose((2, 0, 1))
        #         print(image.shape)
        #         print(gt.shape)
        image = torch.from_numpy(image)  # .cuda(async=True)
        gt = torch.from_numpy(gt)  # .cuda(async=True)
        return {'image': image, 'ground_truth': gt}


def get_train_transform(crop_size=512, stack_bayer=False):
    transform = transforms.Compose(
        [RandomCrop(crop_size, stack_bayer=stack_bayer),
         ToTensor(),
         RandomFlips()]
    )
    return transform


def get_eval_transform(crop_size=512, stack_bayer=False):
    transform = transforms.Compose(
        [RandomCrop(crop_size, stack_bayer=stack_bayer),
         ToTensor()]
    )
    return transform
