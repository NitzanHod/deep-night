from sacred import Experiment

from utils.train_utils.engine_handlers import handlers_ingredient, attach_trainer_events, attach_eval_events
from utils.train_utils.dataloader import dataloader_ingredient, get_dataloaders
from utils.train_utils.optimizer import optimizer_ingredient, get_optimizer
from utils.train_utils.lr_scheduer import lr_scheduler_ingredient, get_lr_scheduler

from utils.sid_engine import engines_ingredient, create_supervised_evaluator, create_supervised_trainer
from utils.train_utils.loss_function import loss_ingredient, get_loss_function
from utils.train_utils.custom_error_metrics import PeakSignalToNoiseRatio, MeanDeepISPError, MeanMSSSIMLuminance
from ignite.metrics import MeanSquaredError, MeanAbsoluteError
import hjson
import os
import torch
from tensorboardX import SummaryWriter
from utils.train_utils.cuda_utils import set_cuda
from torch.nn import L1Loss
from utils.models import NNFactory

CFG_PATH = "cfg/full_cfg_sid.json"

ex = Experiment(ingredients=[engines_ingredient,
                             handlers_ingredient,
                             dataloader_ingredient,
                             optimizer_ingredient,
                             loss_ingredient,
                             lr_scheduler_ingredient])

with open(CFG_PATH) as f:
    ex.add_config(hjson.load(f))


@ex.automain
def run(handlers_cfg, train_cfg):

    # shutil.rmtree(handlers_cfg['dirname'])
    os.makedirs(handlers_cfg['dirname'], exist_ok=True)

    writer = SummaryWriter(handlers_cfg['dirname'])

    train_dataloader, eval_dataloader = get_dataloaders()

    _, device = set_cuda()

    model = NNFactory(train_cfg['model']).to(device)

    if handlers_cfg["multi_gpu"]:
        model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(model)

    scheduler = get_lr_scheduler(optimizer)

    # criterion = loss_function
    # criterion = get_loss_function()
    criterion = L1Loss()

    trainer = create_supervised_trainer(model, optimizer, criterion)
    attach_trainer_events(trainer, model, optimizer=optimizer, scheduler=scheduler)

    metrics = {"L1": MeanAbsoluteError(), "L2": MeanSquaredError(),
               "PSNR": PeakSignalToNoiseRatio(), "DeepISP_Loss": MeanDeepISPError(),
               "Luminance_MS_SSIM": MeanMSSSIMLuminance()}

    train_evaluator = create_supervised_evaluator(model, metrics=metrics)
    attach_eval_events(trainer, train_evaluator, train_dataloader, writer, "Train")

    val_evaluator = create_supervised_evaluator(model, metrics=metrics)
    attach_eval_events(trainer, val_evaluator, eval_dataloader, writer, "Val")

    trainer.run(train_dataloader, train_cfg['max_epochs'])
