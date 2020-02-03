from sacred import Ingredient
import hjson

import utils.train_utils.ms_ssim as ms_ssim
import torch
from utils.rgb2lab import rgb_to_lab

CFG_PATH = "cfg/full_cfg_sid.json"

loss_ingredient = Ingredient('loss')
# handlers_ingredient.add_config(CFG_PATH)

with open(CFG_PATH) as f:
    loss_ingredient.add_config(hjson.load(f))


@loss_ingredient.capture(prefix="train_cfg")
def get_loss_function(loss_fn):
    if loss_fn == "DeepISPLoss":
        return l1_msssim
    if loss_fn == "L1":
        return torch.nn.L1Loss()


def ms_ssim_luminance(y_pred, y, is_rgb):
    if is_rgb:
        lab_input = rgb_to_lab(y_pred)
        lab_output = rgb_to_lab(y)
    else:
        lab_input = y_pred
        lab_output = y
    luminance_input = lab_input[:, 0, :, :].unsqueeze(1)
    luminance_output = lab_output[:, 0, :, :].unsqueeze(1)
    loss = ms_ssim.msssim(luminance_input, luminance_output)
    #print('loss', loss)
    return loss


#  weighted loss of l1 and ms-ssim
def l1_msssim(y_pred, y, alpha=0.5, is_rgb=True):
    #print('shapes', y_pred.size(), y.size())
    if is_rgb:
        lab_input = rgb_to_lab(y_pred)
        lab_output = rgb_to_lab(y)
    else:
        lab_input = y_pred
        lab_output = y
    #print('lab', lab_input.size(), lab_output.size())
    #  l1 loss on lab formatted images
    l1_lab_loss = (1-alpha) * torch.mean(torch.abs(lab_input - lab_output))

    # ms_ssim loss only on luminance channel
    ms_ssim_l_loss = alpha * ms_ssim_luminance(lab_input, lab_output, is_rgb=False)
    #print('loss1', l1_lab_loss)
    #print('loss2', ms_ssim_l_loss)
    return l1_lab_loss + ms_ssim_l_loss

