"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""

import torch
import torchvision

import numpy as np

import inversefed
torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

from collections import defaultdict
import datetime
import time
import os

import argparse

def parser():
    parser = argparse.ArgumentParser(description='Train image classifier model.')

    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)
    
    parser.add_argument('--model', default='ConvNet', type=str, help='Vision model.')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--early_stop', default=5, type=int)

    parser.add_argument('--save_path', required=True, type=str)
    parser.add_argument('--save_epoch', required=True, type=int)

    parser.add_argument('--device', required=True, type=str)

    return parser

args = parser().parse_args()

# Parse training strategy
defs = inversefed.training_strategy('adam')
defs.epochs = args.epochs
defs.validate = 1
defs.early_stop = args.early_stop
defs.save_path = args.save_path
defs.save_epoch = args.save_epoch

if args.batch_size:
    defs.batch_size = args.batch_size

if args.dataset == 'CIFAR10':
    num_classes = 10
elif args.dataset == 'Food101':
    num_classes = 101
elif args.dataset == 'Flowers102':
    num_classes = 102
else:
    raise NotImplementedError()

# 100% reproducibility?
# if args.deterministic:
#     image2graph2vec.utils.set_deterministic()

if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    setup['device'] = args.device

    # Prepare for training

    # Get data:
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs)

    # Find mean and variance by the information of 'transform' class in dataset
    # I know it's dirty to write like that, but I want to modify the code as less as possible
    for t in validloader.dataset.transform.transforms:
        if isinstance(t, torchvision.transforms.Normalize):
            dm = torch.as_tensor(t.mean, **setup)[:, None, None]
            ds = torch.as_tensor(t.std , **setup)[:, None, None]
            break
    else:
        raise Exception("No 'transforms.Normalize' class in dataset")

    model, model_seed = inversefed.construct_model(args.model, num_classes=num_classes, num_channels=3)
    model.to(**setup)

    print('Training the model ...')
    print(repr(defs))
    inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)

    # Sanity check: Validate model accuracy
    training_stats = defaultdict(list)
    inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup, training_stats)
    name, format = loss_fn.metric()
    print(f'Val loss is {training_stats["valid_losses"][-1]:6.4f}, Val {name}: {training_stats["valid_" + name][-1]:{format}}.')

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
