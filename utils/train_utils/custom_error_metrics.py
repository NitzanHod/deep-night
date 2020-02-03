from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from utils.train_utils.loss_function import l1_msssim, ms_ssim_luminance
import numpy as np

class PeakSignalToNoiseRatio(Metric):
    """
    Calculates the PSNR.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_squared_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output

        squared_errors = torch.pow(y_pred - y.view_as(y_pred), 2)
        self._sum_of_squared_errors += torch.sum(squared_errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('PSNR must have at least one example before it can be computed')
        # apparently torch.log10 doesnt work on numbers only tensors
        return 10 * np.log10(1 / (self._sum_of_squared_errors / self._num_examples))


class MeanDeepISPError(Metric):
    """
    Calculates the deepISP error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_l1_msssim_loss = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        self._sum_l1_msssim_loss += l1_msssim(y_pred, y)
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanDeepISPError must have at least one example before it can be computed')
        return self._sum_l1_msssim_loss / self._num_examples


class MeanMSSSIMLuminance(Metric):
    """
    Calculates the deepISP error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_msssim_luminance = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        self._sum_msssim_luminance += ms_ssim_luminance(y_pred, y, is_rgb=True)
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanMSSSIMLuminance must have at least one example before it can be computed')
        return self._sum_msssim_luminance / self._num_examples
