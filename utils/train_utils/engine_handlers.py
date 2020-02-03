from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import EarlyStopping, TerminateOnNan
from ignite.handlers import Checkpoint, DiskSaver
from torchvision import utils
import torch
from sacred import Ingredient
import hjson
import os
import numpy as np


BEFORE_IM_FLAG = True

# Windows Nitzan path
# CFG_PATH = "C:/Users/USER/amral_proj/master/cfg/full_cfg_sid.json"
CFG_PATH = "cfg/full_cfg_sid.json"

handlers_ingredient = Ingredient('handlers')
# handlers_ingredient.add_config(CFG_PATH)

with open(CFG_PATH) as f:
    handlers_ingredient.add_config(hjson.load(f))


@handlers_ingredient.capture(prefix="handlers_cfg")
def attach_trainer_events(trainer, model, optimizer, scheduler, dirname, filename_prefix, save_interval, n_saved, multi_gpu, resume):

    # if output is Nan or inf, stop run
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # add learning rate scheduler
    if scheduler:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    to_save = {'trainer': trainer, 'optimizer': optimizer, 'scheduler': scheduler}

    if multi_gpu:
        to_save['model'] = model.module
    else:
        to_save['model'] = model

    handler = Checkpoint(to_save, save_handler=DiskSaver(dirname=dirname, create_dir=True, require_empty=False),
                         filename_prefix=filename_prefix, n_saved=n_saved)

    if resume:
        checkpoint_file = _get_last_checkpoint_file(dirname)
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            # load_objects replaces the values in to_save to the checkpoint params
            Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=save_interval), handler)

    # add progress bar
    pbar = ProgressBar()
    pbar.attach(trainer)


def _get_last_checkpoint_file(dirname):
    checkpoint_paths = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and f.endswith('.pth')]
    print(checkpoint_paths)
    if len(checkpoint_paths) == 0:
        return None
    else:
        latest_ind = int(np.argmax(np.array([os.path.getctime(os.path.join(dirname, x)) for x in checkpoint_paths])))
        latest_checkpoint_path = os.path.join(dirname, checkpoint_paths[latest_ind])
        return latest_checkpoint_path


def attach_eval_events(trainer, evaluator, eval_dataloader, writer, prefix):

    # add progress bar
    pbar = ProgressBar(desc=prefix)
    pbar.attach(evaluator)

    # add tensorboardX logs
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              create_log_validation_results(evaluator, eval_dataloader, writer, prefix))

    if prefix == "Val":
        add_early_stopping(trainer, evaluator)


def create_log_validation_results(evaluator, dataloader, writer, prefix):
    def log_validation_results(engine):

        epoch = engine.state.epoch

        evaluator.run(dataloader, 1)
        if prefix == "Val":
            global BEFORE_IM_FLAG
            pred, gt = evaluator.state.output

            if BEFORE_IM_FLAG:
                gt_grid = utils.make_grid(gt)
                writer.add_image('Before', gt_grid, epoch)
                BEFORE_IM_FLAG = False

            pred_grid = utils.make_grid(pred)
            writer.add_image('After', pred_grid, epoch)

        metrics_dict = evaluator.state.metrics
        for name, value in metrics_dict.items():
            writer.add_scalars(name, {prefix: value}, engine.state.epoch)
            # print(name, prefix, engine.state.epoch)

    return log_validation_results


@handlers_ingredient.capture(prefix="handlers_cfg")
def add_early_stopping(trainer, evaluator, patience, score_function_name, score_is_loss):

    def score_function(engine):
        value = engine.state.metrics[score_function_name]
        if score_is_loss:
            value *= -1
            # pass
        return value

    early_stopper = EarlyStopping(patience, score_function, trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stopper)
