# Experiment management class, helps injecting sacred configs easily
from utils.train_utils.custom_error_metrics import *
from utils.train_utils.cuda_utils import set_cuda

from utils.train_utils.dataloaders import get_dataloaders
from utils.train_utils.optimizer import get_optimizer
from utils.train_utils.lr_scheduler import get_lr_scheduler
from utils.train_utils.engines import EnginesComponent
from utils.train_utils.handlers import HandlersComponent


from utils.models import NNFactory
from ignite.metrics import MeanSquaredError, MeanAbsoluteError

import os
import torch
import hydra
from tensorboardX import SummaryWriter
from torch.nn import L1Loss


@hydra.main("cfg")
def main_train(cfg):

    def run():
        # shutil.rmtree(handlers_cfg['dirname'])
        # os.makedirs(handlers_cfg['dirname'], exist_ok=True)

        # exp_manager.store_cfg(handlers_cfg['dirname'])

        # writer = SummaryWriter(handlers_cfg['dirname'])
        writer = SummaryWriter(cfg.handlers.dirname)

        # TODO - replace with hydra instantiate?
        train_dataloader, train_eval_dataloader, val_eval_dataloader = get_dataloaders(cfg.dataloader.train_csv_path,
                                                                                       cfg.dataloader.eval_csv_path,
                                                                                       cfg.dataloader.root_dir,
                                                                                       cfg.dataloader.black_level,
                                                                                       cfg.dataloader.stack_bayer,
                                                                                       cfg.dataloader.subset,
                                                                                       cfg.dataloader.crop_size,
                                                                                       cfg.dataloader.batch_size,
                                                                                       cfg.dataloader.shuffle,
                                                                                       cfg.dataloader.num_workers,
                                                                                       cfg.dataloader.pin_memory)

        _, device = set_cuda()

        model = NNFactory(cfg.train.model).to(device)

        if cfg.handlers.multi_gpu:
            model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(model,
                                  cfg.train.optimizer_type,
                                  cfg.train.initial_lr,
                                  cfg.train.momentum,
                                  cfg.train.weight_decay)

        scheduler = get_lr_scheduler(optimizer,
                                     cfg.train.scheduler_type,
                                     cfg.train.step_size,
                                     cfg.train.gamma)

        criterion = L1Loss()

        # metrics = {"L1": MeanAbsoluteError(), "L2": MeanSquaredError(),
        #            "PSNR": PeakSignalToNoiseRatio(), "DeepISP_Loss": MeanDeepISPError(),
        #            "Luminance_MS_SSIM": MeanMSSSIMLuminance(),"Perceptual": MeanPerceptualLoss()}

        metrics = {"L1": MeanAbsoluteError(), "PSNR": PeakSignalToNoiseRatio()}

        engines = EnginesComponent()

        trainer = engines.create_supervised_trainer(model, optimizer, criterion,
                                                    cfg.train.half_precision,
                                                    device)

        handlers = HandlersComponent()
        handlers.attach_trainer_events(trainer, model, optimizer, scheduler,
                                       cfg.handlers.dirname,
                                       cfg.handlers.filename_prefix,
                                       cfg.handlers.save_interval,
                                       cfg.handlers.n_saved,
                                       cfg.handlers.multi_gpu,
                                       cfg.handlers.resume)

        train_evaluator = engines.create_supervised_evaluator(model, metrics=metrics, device=device)
        handlers.attach_eval_events(trainer, model, train_evaluator, train_eval_dataloader, writer, "Eval Train",
                                    cfg.handlers.score_function_name,
                                    cfg.handlers.score_is_loss,
                                    cfg.handlers.patience,
                                    cfg.handlers.dirname)

        val_evaluator = engines.create_supervised_evaluator(model, metrics=metrics, device=device)
        handlers.attach_eval_events(trainer, model, val_evaluator, val_eval_dataloader, writer, "Eval Val",
                                    cfg.handlers.score_function_name,
                                    cfg.handlers.score_is_loss,
                                    cfg.handlers.patience,
                                    cfg.handlers.dirname)

        trainer.run(train_dataloader, cfg.train.max_epochs)

    run()


if __name__ == '__main__':
    main_train()