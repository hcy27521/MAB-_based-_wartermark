import os
from argparse import ArgumentParser

import numpy as np
from train_utils import Trainer, Evaluator
import utils
from easydict import EasyDict
import yaml

from models.vit_small import ViT
from torchvision.models import swin_t
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


def main(cfg):
    # print('\033[1m', '*'*5, f'Restoring for {cfg.trigger_type.split('_')[-1]} triggers, {cfg.trig_lbl[:-3]} labels', '*'*5, '\033[0m')
    utils.set_seed(cfg.seed)
    os.makedirs(os.path.join(cfg.save_dir, cfg.method, '3_retrain'), exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.method, '3_retrain', cfg.save_name)

    train_loader, test_loader, wm_loader, _, ftloader = utils.get_data_from_config(cfg)
    net = utils.get_model_from_config(cfg, len(train_loader.dataset.classes))
    net.load_state_dict(torch.load(os.path.join(cfg.save_dir, cfg.method, '2_finetune', cfg.save_name)))

    # warm up with clean data
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.wd))
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), float(cfg.lr), momentum=0.9, weight_decay=float(cfg.wd))
    
    if cfg.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)
    elif cfg.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=float(cfg.lr_min))
    else:
        lr_scheduler = None
    criterion = nn.CrossEntropyLoss()

    evaluator = Evaluator(net, criterion)
    trainer = Trainer(net, criterion, optimizer, evaluator, train_loader, test_loader, wm_loader, scheduler=lr_scheduler)

    trainer.train(cfg.exp_name, cfg.save_name, cfg.epochs, wandb_project='retrain', eval_pretrain=True, use_wandb=False)

    print(evaluator.eval(test_loader))
    print(evaluator.eval(wm_loader))
    # avg_wm_acc, med_wm_acc = evaluator.eval_robust(wmloader)
    # print(f"Avg WM acc {avg_wm_acc}, Med WM acc {med_wm_acc}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='path to config')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
    main(cfg)