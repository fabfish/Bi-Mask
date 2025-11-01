'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils
import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from data import cifar10
from utils.common import *
from importlib import import_module

import models
import pdb

# Import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
os.environ["WANDB_API_KEY"] = 'b0905a7c1edbfb53372e1fc0ce0717dfe5477326'

# After setting CUDA_VISIBLE_DEVICES, remap GPU indices
original_gpus = args.gpus.copy()
args.gpus = [i for i in range(len(args.gpus))]
# usage: --gpus 0 1 2 3
print(f"Original GPU selection: {original_gpus}")
print(f"Remapped GPU indices: {args.gpus}")
print(f"CUDA_VISIBLE_DEVICES: {visible_gpus_str}")

checkpoint = utils.checkpoint(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger-'+now+'.log'))
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

print("device: ", device)

# Initialize wandb if available and configured
if WANDB_AVAILABLE and args.wandb_project:
    if not args.wandb_name:
        args.wandb_name = f"{args.arch}_{args.mask_mode}_{args.N}_{args.M}_{args.num_epochs}"
    print(f"Initializing wandb with project: {args.wandb_project}, name: {args.wandb_name}")
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# Data
print('==> Loading Data..')

loader = cifar10.Data(args)

from utils.conv_type import NMConv
def apply_post_masks(model):
    for m in model.modules():
        if isinstance(m, NMConv):
            m.post_mask_apply()

def apply_pre_masks(model):
    for m in model.modules():
        if isinstance(m, NMConv):
            m.pre_mask_apply()

def apply_grad_masks(model):
    for m in model.modules():
        if isinstance(m, NMConv):
            m.grad_mask_apply()

def train(model, optimizer, trainLoader, args, epoch):


    model.train()
    losses = utils.AverageMeter(':.4e')

    accurary = utils.AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)

        loss.backward()
        losses.update(loss.item(), inputs.size(0))

        # import pdb; pdb.set_trace()
        if args.mask_mode == "m3":
            apply_pre_masks(model)
        if args.mask_mode == "m5":
            apply_grad_masks(model)
            
        optimizer.step()

        if args.mask_mode == "m4":
            apply_post_masks(model)
        # apply_post_masks(model)
        
        for n,m in model.named_modules():
            if hasattr(m, 'max_iter'):
                m.iter += 1
        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg),  float(accurary.avg), cost_time
                )
            )
            start_time = current_time
            
            # Log to wandb if available
            if WANDB_AVAILABLE and args.wandb_project:
                wandb.log({
                    'epoch': epoch,
                    'batch': batch,
                    'train_loss': float(losses.avg),
                    'train_accuracy': float(accurary.avg),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

def validate(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
        
        # Log validation metrics to wandb if available
        if WANDB_AVAILABLE and args.wandb_project:
            wandb.log({
                'epoch': epoch if 'epoch' in locals() else 0,
                'val_loss': float(losses.avg),
                'val_accuracy': float(accurary.avg)
            })
    
    return accurary.avg

def get_model(args):
    model = models.__dict__[args.arch]().to(device)
    model = model.to(device)
    
    # Print mask mode for each layer
    print(f"\n=== Model Mask Mode Configuration ===")
    print(f"Global mask_mode: {args.mask_mode}")
    print(f"Use random mask: {args.use_random_mask}")
    if args.use_random_mask:
        print(f"Random mask ratio: {args.random_mask_ratio}")
    else:
        print(f"N:M pattern: {args.N}:{args.M}")
    print(f"Layer-wise mask modes:")
    
    from utils.conv_type import NMConv
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, NMConv):
            layer_count += 1
            mask_type = "random" if module.use_random_mask else f"{module.N}:{module.M}"
            print(f"  Layer {layer_count}: {name} -> mask_mode: {module.mask_mode}, mask_type: {mask_type}")
    
    print(f"Total NMConv layers: {layer_count}")
    print("=" * 40)
    
    return model

def get_optimizer(args, model):
    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("alpha" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    else:
        print("please choose sgd or adam")
        
    return optimizer
if __name__ == '__main__':
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print("=> Creating model '{}'".format(args.arch))
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, loader.trainLoader, args, epoch)
        test_acc = validate(model, loader.testLoader)
        scheduler.step()

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        # Log epoch-level metrics to wandb if available
        if WANDB_AVAILABLE and args.wandb_project:
            wandb.log({
                'epoch': epoch,
                'best_accuracy': float(best_acc),
                'current_accuracy': float(test_acc),
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
        }

        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))
    
    # Finish wandb run if available
    if WANDB_AVAILABLE and args.wandb_project:
        wandb.finish()

