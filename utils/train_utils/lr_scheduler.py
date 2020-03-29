from torch import optim

from ignite.contrib.handlers.param_scheduler import LRScheduler
from utils.cfg_utils import ExperimentManager

lr_scheduler_ingredient = ExperimentManager().get_ingredient('lr_scheduler')


class ReduceLROnPlateauScheduler(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, train_evaluator, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        super(ReduceLROnPlateauScheduler, self).__init__(optimizer, mode, factor, patience,
                                                         verbose, threshold, threshold_mode, cooldown, min_lr, eps)

        self.score_function_name = _get_score_function_name()
        self.evaluator = train_evaluator

    def __call__(self, engine):
        if self.evaluator.state is not None:
            self.step(self.evaluator.state.metrics[self.score_function_name])


@lr_scheduler_ingredient.capture(prefix="train_cfg")
def get_lr_scheduler(optimizer, train_evaluator,  scheduler_type, step_size=2000, gamma=0, factor=0.1, patience=10, verbose=False):
    if scheduler_type == "steps":
        step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler = LRScheduler(step_scheduler)

    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateauScheduler(optimizer, train_evaluator, factor=factor, patience=patience, verbose=verbose)
    else:
        raise ValueError("scheduler type", scheduler_type, "is not supported")
    return scheduler


@lr_scheduler_ingredient.capture(prefix="handlers_cfg")
def _get_score_function_name(score_function_name):
    return score_function_name
