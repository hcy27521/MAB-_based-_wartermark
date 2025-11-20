import os
from argparse import ArgumentParser

import numpy as np
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



def main(cfg):
    # print('\033[1m', '*'*5, f'Restoring for {cfg.trigger_type.split('_')[-1]} triggers, {cfg.trig_lbl[:-3]} labels', '*'*5, '\033[0m')
    utils.set_seed(cfg.seed)
    # 1. 组合目录和文件名
    full_save_name = os.path.join(cfg.save_dir, cfg.save_name)
    # 2. 确保目录存在 (我们只需要目录部分)
    save_dir_only = os.path.dirname(full_save_name)
    os.makedirs(save_dir_only, exist_ok=True) # 这行应该确保 'checkpoints' 目录存在   

    #save_path = cfg.save_name.rsplit('/', 1)[0]
    #os.makedirs(save_path, exist_ok=True)

    train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, _ = utils.get_data_from_config(cfg, with_mark=True)
    if cfg.method == 'app':
        train_loader_app, test_loader_app, wm_loader_app, train_wm_loader_app, ft_loader_app = utils.get_data_from_config(cfg, with_mark=True)
    net = utils.get_model_from_config(cfg, len(train_loader.dataset.classes))

    # warm up with clean data
    def _get_init_value(config_value):
        if isinstance(config_value, list):
            # 检查列表是否为空，避免 IndexError
            if not config_value:
                raise ValueError("Configuration list is empty.")
            return config_value[0]
        return config_value

    # 1. 提取所有需要的值，并在后续代码中使用这些新变量
    # 优化器名称 (需要转小写)
    optimizer_name = _get_init_value(cfg.optimizer).lower() 
    # 学习率 (需要转 float)
    lr_value = float(_get_init_value(cfg.lr))
    # 权重衰减 (需要转 float)
    wd_value = float(_get_init_value(cfg.wd))
    # 调度器名称 (需要转小写)
    scheduler_name = _get_init_value(cfg.scheduler).lower()
    # 调度器步长 (需要转 int)
    scheduler_step = int(_get_init_value(cfg.scheduler_step))
    # 调度器 gamma (需要转 float)
    scheduler_gamma = float(_get_init_value(cfg.scheduler_gamma))
    # 最小学习率 (需要转 float)
    lr_min_value = float(_get_init_value(cfg.lr_min))
    # 训练 Epochs (需要转 int)
    epochs_value = int(_get_init_value(cfg.epochs))

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr_value, weight_decay=wd_value)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr_value, momentum=0.9, weight_decay=wd_value)
    
    if cfg.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    elif cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_value, eta_min=lr_min_value)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    if cfg.method == 'app':
        evaluator = Evaluator(net, criterion, mark=True)
        trainer = APPTrainer(net, criterion, optimizer, evaluator, train_wm_loader_app, test_loader_app, wm_loader_app, scheduler=scheduler, batchsize_p=cfg.batchsize_wm, batchsize_c=cfg.batchsize_c, app_eps=cfg.app_eps, app_alpha=cfg.app_alpha)
    elif cfg.method == 'adi':
        evaluator = Evaluator(net, criterion)
        trainer = Trainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
    elif cfg.method == 'certified':
        evaluator = Evaluator(net, criterion)
        trainer = CertifiedTrainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
    elif cfg.method == 'rowback':
        evaluator = Evaluator(net, criterion)
        trainer = ROWBACKTrainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
    elif cfg.method == 'MAB':
        evaluator = Evaluator(net, criterion)
        trainer = MABTrainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
    trainer.train(None, full_save_name, epochs_value)
    if cfg.method == 'rowback':
        frozen_layers =[trainer.net.conv1, trainer.net.bn1, trainer.net.layer1, trainer.net.layer2, trainer.net.layer3, trainer.net.layer4]
        trainer.train_freeze(None, cfg.save_name, epochs_value, train_wm_loader, frozen_layers=frozen_layers, wandb_project=None, eval_robust=False, eval_pretrain=False, use_wandb=False, save_every=5)
    

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