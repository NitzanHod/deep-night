import utils.train_utils.ms_ssim as ms_ssim
import torch
import torchvision
from utils.rgb2lab import rgb_to_lab


class LossFunctionComponent:

    def __init__(self, ingredient):
        @ingredient.capture(prefix="train_cfg")
        def get_loss_function(loss_fn):
            loss_fn = str(loss_fn).strip().lower()
            if loss_fn == "deepisploss":
                return l1_msssim
            if loss_fn == "l1":
                return torch.nn.L1Loss()
            if loss_fn == "perceptual":
                return VGGPerceptualLoss()

        self.methods = [get_loss_function]

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
    # print('loss', loss)
    return loss


#  weighted loss of l1 and ms-ssim
def l1_msssim(y_pred, y, alpha=0.5, is_lab=False):
    # print('shapes', y_pred.size(), y.size())
    if is_lab:
        lab_input = y_pred
        lab_output = y
    else:
        lab_input = rgb_to_lab(y_pred)
        lab_output = rgb_to_lab(y)
    # print('lab', lab_input.size(), lab_output.size())
    #  l1 loss on lab formatted images
    l1_lab_loss = (1 - alpha) * torch.mean(torch.abs(lab_input - lab_output))

    # ms_ssim loss only on luminance channel
    ms_ssim_l_loss = alpha * ms_ssim_luminance(lab_input, lab_output, is_rgb=False)
    # print('loss1', l1_lab_loss)
    # print('loss2', ms_ssim_l_loss)
    return l1_lab_loss + ms_ssim_l_loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device='cuda:0', resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        features = torchvision.models.vgg16(pretrained=True).to(device=device).features

        blocks.append(features[:4].eval())
        blocks.append(features[4:9].eval())
        blocks.append(features[9:16].eval())
        blocks.append(features[16:23].eval())

        del features

        for bl in blocks:
            for p in bl:
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss
