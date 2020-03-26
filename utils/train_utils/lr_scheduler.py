from torch import optim
from ignite.contrib.handlers.param_scheduler import LRScheduler
from utils.cfg_utils import ExperimentManager

lr_scheduler_ingredient = ExperimentManager().get_ingredient('lr_scheduler')


@lr_scheduler_ingredient.capture(prefix="train_cfg")
def get_lr_scheduler(optimizer, scheduler_type, step_size, gamma=0):
    if scheduler_type == "steps":
        step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler = LRScheduler(step_scheduler)
    else:
        raise ValueError("scheduler type", scheduler_type, "is not supported")
    return scheduler

