import os
from argparse import ArgumentParser

import numpy as np
import data_utils
from train_utils import *
import utils
from easydict import EasyDict
import yaml

from models.vit_small import ViT
from torchvision.models import swin_t
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import plot_trajectory, plot_surface, plot_2D



def main(cfg, exp_type):
    print('='*5, 'Generating loss landscape', '='*5)
    if exp_type == 'ftal':
        cfg.extraction = False
    elif exp_type == 'extraction':
        cfg.extraction = True
    args = plot_trajectory.plot(cfg)
    args = plot_surface.plot(cfg, args)
    plot_2D.plot(args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='path to config')
    parser.add_argument('exp_type', type=str, help='types of experiments, can be [ftal, extraction, both]')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
        if cfg.method == 'ewe':
            cfg.ewe = True
            cfg.regenerate = False
    main(cfg, args.exp_type)