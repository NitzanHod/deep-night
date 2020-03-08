from torch import optim


def get_optimizer(model, optimizer_type, initial_lr, momentum, weight_decay):
    initial_lr = float(initial_lr)  # handles '1e-4' format
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum)
    else:
        raise ValueError("Optimizer type", optimizer_type, "is not supported")
    return optimizer
