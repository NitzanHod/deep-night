import torch


def select_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def select_dtype():
    return torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def set_cuda():
    return select_dtype(), select_device()
