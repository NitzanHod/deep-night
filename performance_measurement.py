from utils.models import NNFactory
from utils.train_utils.cuda_utils import set_cuda
import torch
import matplotlib.pyplot as plt
import time,os
import numpy as np
import rawpy
from utils.train_utils.dataloaders import raw2np, pack_raw


dtype, device = set_cuda()
sample = torch.rand(1, 4, 1024, 512).to(device).type(dtype)

configuration_path = 'cfg/model_cfg/orig_u_net.cfg'
image_path = 'dataset/Miniset/short/00001_00_0.1s.ARW'

# load an image to a tensor
raw = rawpy.imread(image_path)
packed =pack_raw(raw2np(raw, black_level=512))
print(packed.shape)

sample = torch.tensor(packed).permute(2,0,1).unsqueeze(0)
print(sample.size())
model = NNFactory(configuration_path)

if configuration_path ==   'cfg/model_cfg/orig_u_net.cfg':
    model_dict = model.state_dict()
    loaded_state_dict = torch.load('final.pt')
    for model_key, loaded_value in zip(model_dict.keys(), loaded_state_dict.values()):
        model_dict[model_key] = loaded_value
    model.load_state_dict(model_dict)

model.eval().to(device).type(dtype)
res = model(sample)
print(res.size())
res = res.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
print(res.shape)
print(np.max(res))
print(np.min(res))

# res = (res - np.min(res) )/ (np.max(res) - np.min(res))

print(np.max(res))
print(np.min(res))

plt.imshow(res)
plt.show()
"""
time_list = []
for _ in np.arange(5):
    tic = time.time()

    res = model(sample)

    time_list.append(time.time() - tic)
"""

# res_list = np.array(model.time_list[1:])
# print('total mu:', res_list.mean())
# print('total sigma:', res_list.std())
