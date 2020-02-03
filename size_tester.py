import torch
"""
checking if del actually frees memory on gpu, since it isn't showing it on nvidia-smi.
"""

# will crush at ~27
# tensors = []
# for _ in range(30):
#     print(len(tensors))
#     tensors.append(torch.zeros(10000, 10000).cuda())


# won't crush
tensors = []
for _ in range(30):
    print(len(tensors))
    tensors.append(torch.zeros(10000, 10000).cuda())
    del tensors[len(tensors) - 1]