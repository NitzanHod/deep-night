from sacred import Ingredient
from torch import optim
import hjson

CFG_PATH = "cfg/full_cfg_sid.json"

optimizer_ingredient = Ingredient('optimizer')
# handlers_ingredient.add_config(CFG_PATH)
with open(CFG_PATH) as f:
    optimizer_ingredient.add_config(hjson.load(f))


@optimizer_ingredient.capture(prefix="train_cfg")
def get_optimizer(model, optimizer_type, initial_lr, momentum=0.9, weight_decay=0):
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum)
    else:
        raise ValueError("Optimizer type", optimizer_type, "is not supported")
    return optimizer
