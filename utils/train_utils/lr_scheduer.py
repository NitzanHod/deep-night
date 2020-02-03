from sacred import Ingredient
from torch import optim
from ignite.contrib.handlers.param_scheduler import LRScheduler
import hjson

CFG_PATH = "cfg/full_cfg_sid.json"

lr_scheduler_ingredient = Ingredient('lr_scheduler')
# handlers_ingredient.add_config(CFG_PATH)
with open(CFG_PATH) as f:
    lr_scheduler_ingredient.add_config(hjson.load(f))


@lr_scheduler_ingredient.capture(prefix="train_cfg")
def get_lr_scheduler(optimizer, scheduler_type, step_size, gamma=0):
    if scheduler_type == "steps":
        step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler = LRScheduler(step_scheduler)
    else:
        raise ValueError("scheduler type", scheduler_type, "is not supported")
    return scheduler
