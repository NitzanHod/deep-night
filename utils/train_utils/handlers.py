from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import EarlyStopping, TerminateOnNan, global_step_from_engine
from ignite.handlers import Checkpoint, DiskSaver
from torchvision import utils
import torch
import os
import numpy as np

from utils.display_utils import sprint


class HandlersComponent:
    BEFORE_IM_FLAG = True

    def __init__(self, ingredient):

        @ingredient.capture(prefix="handlers_cfg")
        def attach_trainer_events(trainer, model, optimizer, scheduler, dirname, filename_prefix, save_interval,
                                  n_saved, multi_gpu, resume):

            # if output is Nan or inf, stop run
            trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

            # add learning rate scheduler
            if scheduler:
                trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

            to_save, handler = _prepare_checkpoint(trainer, model, optimizer, scheduler)

            if resume:
                resume_last_checkpoint(to_save)

            trainer.add_event_handler(Events.EPOCH_COMPLETED(every=save_interval), handler)

            # add progress bar
            pbar = ProgressBar()
            pbar.attach(trainer)

        @ingredient.capture(prefix="handlers_cfg")
        def resume_last_checkpoint(to_save, dirname, filename_prefix):
            checkpoint_file = _get_last_checkpoint_file(dirname, filename_prefix)
            if checkpoint_file is not None:
                checkpoint = torch.load(checkpoint_file)
                # load_objects replaces the values in to_save to the checkpoint params
                Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

        @ingredient.capture(prefix="handlers_cfg")
        def _prepare_checkpoint(trainer, model, optimizer, scheduler, dirname, filename_prefix, n_saved, multi_gpu):
            to_save = {'trainer': trainer, 'optimizer': optimizer, 'scheduler': scheduler}

            if multi_gpu:
                to_save['model'] = model.module
            else:
                to_save['model'] = model

            handler = Checkpoint(to_save, save_handler=DiskSaver(dirname=dirname, create_dir=True, require_empty=False),
                                 filename_prefix=filename_prefix, n_saved=n_saved)

            return to_save, handler

        def _get_last_checkpoint_file(dirname, filename_prefix):
            checkpoint_paths = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))
                                and f.endswith('.pth') and f.startswith(filename_prefix)]
            if len(checkpoint_paths) == 0:
                return None
            else:
                latest_ind = int(
                    np.argmax(np.array([os.path.getctime(os.path.join(dirname, x)) for x in checkpoint_paths])))
                latest_checkpoint_path = os.path.join(dirname, checkpoint_paths[latest_ind])
                try:
                    torch.load(latest_checkpoint_path)
                except EOFError:
                    # training stopped mid-writing, should use one before last
                    # if it is the first epoch, start from scratch (returns None)
                    if latest_ind == 0:
                        return None
                    else:
                        latest_checkpoint_path = os.path.join(dirname, checkpoint_paths[latest_ind - 1])
                return latest_checkpoint_path

        @ingredient.capture(prefix="handlers_cfg")
        def attach_eval_events(trainer, model, evaluator, eval_dataloader, writer,
                               prefix, dirname, score_function_name, score_is_loss):

            # add progress bar
            pbar = ProgressBar(desc=prefix)
            pbar.attach(evaluator)

            # add tensorboardX logs
            trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                      create_log_validation_results(evaluator, eval_dataloader, writer, prefix))

            if prefix == "Eval Val":
                def score_function(engine):
                    value = engine.state.metrics[score_function_name]
                    if score_is_loss:
                        value *= -1
                        # pass
                    return value

                add_early_stopping(trainer, evaluator, score_function)

                add_best_save(trainer, evaluator, model, score_function)

        @ingredient.capture(prefix="handlers_cfg")
        def add_best_save(trainer, evaluator, model, score_function, dirname, score_function_name):
            to_save = {'model': model}
            handler = Checkpoint(to_save, DiskSaver(dirname=dirname, create_dir=True, require_empty=False), n_saved=1,
                                 filename_prefix='best', score_function=score_function, score_name=score_function_name,
                                 global_step_transform=global_step_from_engine(trainer))

            evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

        def create_log_validation_results(evaluator, dataloader, writer, prefix):
            def log_validation_results(engine):

                epoch = engine.state.epoch

                evaluator.run(dataloader, 1)
                if prefix == "Eval Val":
                    pred, gt = evaluator.state.output
                    # sprint(pred, "Pred")
                    # sprint(gt, "GT")
                    if HandlersComponent.BEFORE_IM_FLAG:
                        gt_grid = utils.make_grid(gt)
                        writer.add_image('GT', gt_grid, epoch)
                        HandlersComponent.BEFORE_IM_FLAG = False

                    pred_grid = utils.make_grid(pred)
                    writer.add_image('Pred', pred_grid, epoch)

                metrics_dict = evaluator.state.metrics
                for name, value in metrics_dict.items():
                    writer.add_scalars(name, {prefix: value}, engine.state.epoch)
                    # print(name, prefix, engine.state.epoch)

            return log_validation_results

        @ingredient.capture(prefix="handlers_cfg")
        def add_early_stopping(trainer, evaluator, score_function, patience):
            early_stopper = EarlyStopping(patience, score_function, trainer)
            evaluator.add_event_handler(Events.COMPLETED, early_stopper)

        self.methods = [attach_trainer_events,
                        resume_last_checkpoint,
                        attach_eval_events,
                        add_best_save,
                        create_log_validation_results,
                        add_early_stopping]
