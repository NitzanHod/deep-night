from utils.models import NNFactory
from utils.train_utils.cuda_utils import set_cuda
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

dtype, device = set_cuda()
height, width = 512, 512
sample_size = str(height) + '*' + str(width)

mean_t = {}
cutoff = 20
num_iterations = 50

type_results = {}
paths = ['cfg/model_cfg/deepisp_rgb.cfg',
         'cfg/model_cfg/deepisp_reg_conv_ll15_hl3.cfg',
         'cfg/model_cfg/deepisp_packed_bef_transform.cfg',
         'cfg/model_cfg/orig_u_net.cfg']

names = ['RGB', 'Full', 'Pixel Bef', 'U Net']
colors = ['r', 'b', 'g', 'k', 'hotpink']
markers = ['x', 'o', '+', '*', '^']
models = enumerate([NNFactory(x) for x in paths])
for model_ind, model in models:

    print('Running model -', names[model_ind])

    model.eval().to(device).type(dtype)
    time_list = []
    module_times = None

    for iteration in np.arange(num_iterations):

        if model_ind in [1, 2, 3]:
            sample = torch.rand(1, 4, height // 2, width // 2).to(device).type(dtype)
        elif model_ind in []:
            sample = torch.rand(1, 1, height, width).to(device).type(dtype)
        elif model_ind in [0]:
            sample = torch.rand(1, 3, height, width).to(device).type(dtype)
        #print(iteration, '...')
        #torch.cuda.synchronize()

        tic = time.time()
        # with torch.no_grad():
        res = model(sample)

        curr_module_times = torch.Tensor(model.time_list)
        #torch.cuda.synchronize()

        toc = time.time() - tic

        if iteration == 0:
            module_times = torch.Tensor(curr_module_times).unsqueeze(0)
        else:
            module_times = torch.cat((module_times, curr_module_times.unsqueeze(0)), dim=0)

        time_list.append(toc)

    total_time_per_iteration = torch.tensor(time_list)

    # TODO: explain time overhead spent between layers (takes most of the runtime)
    # should be num iterations, num layers

    all_layers_per_iteration = torch.sum(module_times, dim=1)

    between_layer_difference = total_time_per_iteration - all_layers_per_iteration
    print('mean overhead: ', "%1.5f" % float(torch.mean(total_time_per_iteration[cutoff:]).item())+'-'+
          "%1.5f" % float(torch.mean(all_layers_per_iteration[cutoff:]).item())+'='+
          "%1.5f" % float(torch.mean(between_layer_difference[cutoff:]).item()))
    # displaying the model layers performance history
    for i, module_def in enumerate(model.module_defs):
        if True:  # module_def['type'] in ['bilinear_interpolation', 'quadratic_transform']:
            plt.scatter(np.arange(num_iterations - cutoff), module_times[cutoff:, i],
                        label=names[model_ind] + ' ' + module_def['type'] + ' ' + str(i),
                        marker=markers[model_ind])  # , c=colors[model_ind])

    mean_t[model_ind] = ("%1.4f" % np.mean(time_list[cutoff:]))
    # print('Mean time', mean_t[model_ind])

title = 'Comparing Pixel Shuffle Models to U Net'
for i in np.arange(len(paths)):
    title += names[i]+' '+mean_t[i]+'\n'

plt.title(title)
plt.ylim(-0.00001, 0.001)
plt.xlabel('#Iteration')
plt.ylabel('Time (s)')
plt.legend(ncol=2)
plt.show()

# res_list = np.array(model.time_list[1:])
# print('total mu:', res_list.mean())
# print('total sigma:', res_list.std())
