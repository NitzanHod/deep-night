from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import torch

from utils.parse_config import parse_model_cfg
from utils.train_utils.cuda_utils import set_cuda


import time

ONNX_EXPORT = False


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            groups = 1
            module = nn.Conv2d(in_channels=output_filters[-1],
                               out_channels=filters,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=pad,
                               groups=groups,
                               bias=not bn)

            modules.add_module('conv_%d' % i, module)
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0)
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'avgpool':
            kernel_size = int(module_def['size'])
            avgpool = nn.AvgPool2d(kernel_size=kernel_size)
            modules.add_module('avgpool_%d' % i, avgpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        # route concatenates channels
        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        # shortcut sums channels
        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'deconvolutional':
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            groups = 1
            module = nn.ConvTranspose2d(in_channels=output_filters[-1],
                                        out_channels=filters,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=pad,
                                        groups=groups,
                                        bias=True)

            modules.add_module('deconv_%d' % i, module)

        elif module_def['type'] == 'pixel_shuffle':
            filters = 3
            upscale_factor = int(module_def['upscale_factor'])
            pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
            modules.add_module('pixel_shuffle_%d' % i, pixel_shuffle)

        if 'activation' in module_def.keys():
            if module_def['activation'] == 'leaky':
                slope = float(module_def['slope'])
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(slope))
            elif module_def['activation'] == 'relu':
                modules.add_module('relu_%d' % i, nn.ReLU())
            elif module_def['activation'] == 'relu6':
                modules.add_module('relu6_%d' % i, nn.ReLU6())
            elif module_def['activation'] == 'htanh':
                modules.add_module('htanh_%d' % i, nn.Hardtanh())
            elif module_def['activation'] == 'tanh':
                modules.add_module('tanh_%d' % i, nn.Tanh())
            elif module_def['activation'] == 'none':
                pass
            else:
                NotImplementedError(f'Activation function {module_def["activation"]} unrecognized')

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


def get_non_linearity(non_linearity):
    if non_linearity == 'relu6':
        return nn.ReLU6()
    elif non_linearity == 'relu':
        return nn.ReLU()
    elif non_linearity == 'tanh':
        return nn.Tanh()
    elif non_linearity == 'htanh':
        return nn.Hardtanh()
    else:
        raise NotImplementedError


class NNFactory(nn.Module):

    """YOLOv3 object detection model"""
    def __init__(self, cfg_path, img_size=416):
        super(NNFactory, self).__init__()
        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'xy', 'wh', 'conf', 'cls', 'nT']
        self.losses = []

        # since this is the first layer we set the type here
        dtype, device = set_cuda()
        self.dtype = dtype
        self.device = device

        self.time_list = []

        self.filter_ranks = {}
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x, targets=None):
        self.time_list = []

        x = x.to(self.device).type(self.dtype)
        self.losses = defaultdict(float)
        is_training = targets is not None
        layer_outputs = []

        output = []

        tic = time.time()
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):

            mtype = module_def['type']
            mod_tic = time.time()
            # print(mtype, x.size())
            if mtype in ['upsample', 'maxpool', 'avgpool', 'linear', 'rgb_lab_converter', 'noop','convolutional',
                         'mean', 'deepisp_block', 'pixel_shuffle','linear_packed', 'deconvolutional']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    if 'start' in module_def and 'end' in module_def and len(module_def['layers'].split(',')) == 1:
                        x = layer_outputs[layer_i[0]][:, int(module_def['start']):int(module_def['end']), :, :]
                    else:
                        x = layer_outputs[layer_i[0]]
                else:
                    # print('Routing...')
                    # [print(layer_outputs[i].size()) for i in layer_i]
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)

            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            if 'output' in module_def.keys():
                if int(module_def['output']) == 1:
                    output.append(x)

            layer_outputs.append(x)

            # print(mtype, route_counters, [i for i, x in enumerate(layer_outputs) if x is not None])

            mod_time = time.time()-mod_tic
            self.time_list.append(mod_time)

        # print(curr_time - tic)
        if is_training:
            self.losses['nT'] /= 3
        if ONNX_EXPORT:
            output = torch.cat(output, 1)  # merge the 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            return output[5:85].t(), output[:4].t()  # ONNX scores, boxes
        # print('Odd time difference, should be less than 10^-4', (time.time() - tic) - sum(self.time_list))

        return sum(output) if is_training else torch.cat(output, 1)