import torch
from ignite.engine.engine import Engine
from ignite._utils import convert_tensor
from utils.train_utils.cuda_utils import set_cuda
from utils.cfg_utils import ExperimentManager

engines_ingredient = ExperimentManager().get_ingredient('engines')


def _prepare_batch(batch, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    _, device = set_cuda()
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


@engines_ingredient.capture(prefix="train_cfg")
def create_supervised_trainer(model, optimizer, loss_fn, half_precision,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (Callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)

    # half_precision = amp.init(fp16_train)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, non_blocking=non_blocking)
        y_pred = model(x)
        # print(y_pred.size(), y.size())
        loss = loss_fn(y_pred, y)
        # with half_precision.scale_loss(loss, optimizer) as scaled_loss:
        # scaled_loss.backward()
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics={},
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (Callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.

    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        # with torch.no_grad():
        x, y = prepare_batch(batch, non_blocking=non_blocking)
        y_pred = model(x)
        return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
