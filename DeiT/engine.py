# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from timm.models.vision_transformer import NMConv

def apply_post_masks(model):
    """
    在 optimizer.step() 之后调用，遍历所有 NMConv，对 mask_mode == 'm4_post' 或需要 post 投影的层执行权重投影。
    """
    for m in model.modules():
        if isinstance(m, NMConv):
            # respect per-layer enable flag if present
            if getattr(m, 'mask_enabled', True):
                if getattr(m, 'mask_mode', None) == 'm4' or getattr(m, 'mask_mode', None) == 'm4_post':
                    # m4: apply after optimizer.step
                    if hasattr(m, 'post_mask_apply'):
                        m.post_mask_apply()

def apply_pre_masks(model):
    """
    在 backward 之前应用的 mask（用于 m3 或其他需要在反向传播前修改权重的模式）
    """
    for m in model.modules():
        if isinstance(m, NMConv):
            if getattr(m, 'mask_enabled', True) and getattr(m, 'mask_mode', None) == 'm3':
                # m3: apply pre mask projection before backward()
                if hasattr(m, 'pre_mask_apply'):
                    m.pre_mask_apply()

def apply_grad_masks(model):
    """
    在 optimizer.step() 之前应用梯度掩码（用于 m5）
    """
    for m in model.modules():
        if isinstance(m, NMConv):
            if getattr(m, 'mask_enabled', True) and getattr(m, 'mask_mode', None) == 'm5':
                if hasattr(m, 'grad_mask_apply'):
                    m.grad_mask_apply()
                
def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        #with torch.cuda.amp.autocast():
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # 如果当前模型层有需要在 backward 前应用的投影（例如 m3），先应用
        apply_pre_masks(model)

        # backward
        loss.backward()

        # 如果需要在 optimizer.step() 前对梯度/权重进行掩码（例如 m5），应用它
        apply_grad_masks(model)

        optimizer.step()

        # 在 step 后应用 post-mask（例如 m4）
        apply_post_masks(model)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}