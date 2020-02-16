# Experiment management class, helps injecting sacred configs easily
from utils.cfg_utils import ExperimentManager
from utils.train_utils.custom_error_metrics import PeakSignalToNoiseRatio, MeanDeepISPError, MeanMSSSIMLuminance
from utils.train_utils.cuda_utils import set_cuda

from utils.models import NNFactory
from ignite.metrics import MeanSquaredError, MeanAbsoluteError

import os
import torch

from tensorboardX import SummaryWriter
from torch.nn import L1Loss


if __name__ == '__main__':
    exp_manager = ExperimentManager('test')
    component_names = ['optimizer', 'engines', 'handlers', 'dataloaders', 'lr_scheduler', 'loss_function']
    ex = exp_manager.prepare_run(component_names=component_names)

    @ex.automain
    def run(handlers_cfg, train_cfg):

        # shutil.rmtree(handlers_cfg['dirname'])
        os.makedirs(handlers_cfg['dirname'], exist_ok=True)

        writer = SummaryWriter(handlers_cfg['dirname'])

        train_dataloader, eval_dataloader = exp_manager.get_dataloaders()

        _, device = set_cuda()

        model = NNFactory(train_cfg['model']).to(device)

        if handlers_cfg["multi_gpu"]:
            model = torch.nn.DataParallel(model)

        optimizer = exp_manager.get_optimizer(model)

        scheduler = exp_manager.get_lr_scheduler(optimizer)

        criterion = L1Loss()

        metrics = {"L1": MeanAbsoluteError(), "L2": MeanSquaredError(),
                   "PSNR": PeakSignalToNoiseRatio(), "DeepISP_Loss": MeanDeepISPError(),
                   "Luminance_MS_SSIM": MeanMSSSIMLuminance()}

        trainer = exp_manager.create_supervised_trainer(model, optimizer, criterion)
        exp_manager.attach_trainer_events(trainer, model, optimizer=optimizer, scheduler=scheduler)

        train_evaluator = exp_manager.create_supervised_evaluator(model, metrics=metrics)
        exp_manager.attach_eval_events(trainer, model, train_evaluator, train_dataloader, writer, "Eval Train")

        val_evaluator = exp_manager.create_supervised_evaluator(model, metrics=metrics)
        exp_manager.attach_eval_events(trainer, model, val_evaluator, eval_dataloader, writer, "Eval Val")

        trainer.run(train_dataloader, train_cfg['max_epochs'])