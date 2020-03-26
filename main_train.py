# Experiment management class, helps injecting sacred configs easily
from sacred import Experiment

from utils.cfg_utils import ExperimentManager
from utils.train_utils.cuda_utils import set_cuda
from utils.train_utils.model_utils import initialize_weights

from utils.models import NNFactory
from ignite.metrics import MeanSquaredError, MeanAbsoluteError

import os
import torch

from tensorboardX import SummaryWriter
from torch.nn import L1Loss


if __name__ == '__main__':
    singleton = ExperimentManager('cfg/train_finetune.yml')

    # import modules that are dependent on the singleton
    from utils.train_utils.custom_error_metrics import *
    from utils.train_utils.handlers import handlers_ingredient, attach_trainer_events, attach_eval_events
    from utils.train_utils.dataloaders import dataloader_ingredient, get_dataloaders
    from utils.train_utils.optimizer import optimizer_ingredient, get_optimizer
    from utils.train_utils.lr_scheduler import lr_scheduler_ingredient, get_lr_scheduler
    from utils.train_utils.engines import engines_ingredient, create_supervised_evaluator, create_supervised_trainer
    from utils.train_utils.loss_function import loss_ingredient, get_loss_function

    ex = Experiment(ingredients=[engines_ingredient,
                                 handlers_ingredient,
                                 dataloader_ingredient,
                                 optimizer_ingredient,
                                 loss_ingredient,
                                 lr_scheduler_ingredient])

    ex.add_config(singleton.config)

    @ex.automain
    def run(handlers_cfg, train_cfg):

        # shutil.rmtree(handlers_cfg['dirname'])
        os.makedirs(handlers_cfg['dirname'], exist_ok=True)

        ExperimentManager().store_cfg(handlers_cfg['dirname'])

        writer = SummaryWriter(handlers_cfg['dirname'])

        train_dataloader,  train_eval_dataloader, val_eval_dataloader = get_dataloaders()

        _, device = set_cuda()

        model = NNFactory(train_cfg['model']).to(device)

        if 'weights' in train_cfg.keys() and os.path.isfile(train_cfg['weights']):
            model = initialize_weights(model, train_cfg['weights'])

        if handlers_cfg["multi_gpu"]:
            model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(model)

        scheduler = get_lr_scheduler(optimizer)

        criterion = get_loss_function()

        # metrics = {"L1": MeanAbsoluteError(), "L2": MeanSquaredError(),
        #            "PSNR": PeakSignalToNoiseRatio(), "DeepISP_Loss": MeanDeepISPError(),
        #            "Luminance_MS_SSIM": MeanMSSSIMLuminance(),"Perceptual": MeanPerceptualLoss()}

        metrics = {"L1": MeanAbsoluteError(), "PSNR": PeakSignalToNoiseRatio()}

        trainer = create_supervised_trainer(model, optimizer, criterion)
        attach_trainer_events(trainer, model, optimizer=optimizer, scheduler=scheduler)

        train_evaluator = create_supervised_evaluator(model, metrics=metrics)
        attach_eval_events(trainer, model, train_evaluator, train_eval_dataloader, writer, "Eval Train")

        val_evaluator = create_supervised_evaluator(model, metrics=metrics)
        attach_eval_events(trainer, model, val_evaluator, val_eval_dataloader, writer, "Eval Val")

        trainer.run(train_dataloader, train_cfg['max_epochs'])