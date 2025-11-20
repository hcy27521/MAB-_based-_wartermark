import os
import wandb
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


def get_optims(net, cfg, phase_idx):
    if cfg.optimizer[phase_idx] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.lr[phase_idx]), weight_decay=float(cfg.wd[phase_idx]))
    elif cfg.optimizer[phase_idx] == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), float(cfg.lr[phase_idx]), momentum=0.9, weight_decay=float(cfg.wd[phase_idx]))

    scheduler = None
    if cfg.scheduler[phase_idx] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step[phase_idx], gamma=cfg.scheduler_gamma[phase_idx])
    elif cfg.scheduler[phase_idx] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler_step[phase_idx], gamma=cfg.scheduler_gamma[phase_idx])
    elif cfg.scheduler[phase_idx] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs[phase_idx], eta_min=float(cfg.lr_min[phase_idx]))
    return optimizer, scheduler

def main(cfg):
    utils.set_seed(cfg.seed)

    # data setup
    classes = []
    if cfg.method == 'ewe':
        train_loader, test_loader, wm_loader, wm_loader_mix, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True, ewe=True)
        classes = train_loader.dataset.classes
    elif cfg.method == 'app':
        train_loader_app, test_loader_app, wm_loader_app, train_wm_loader_app, ft_loader_app, train_ft_loader_app = utils.get_data_from_config(cfg, with_mark=True, train_ft_mix=True)
        classes = train_loader_app.dataset.classes
    elif cfg.method == 'na':
        train_loader, test_loader = utils.get_data_from_config(cfg)
        classes = test_loader.dataset.classes
    else:
        train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True)
        classes = train_loader.dataset.classes

        
    net = utils.get_model_from_config(cfg, len(classes))
    print(f'\033[1m***** Experiments for WM method {cfg.method}, {cfg.trigger_type.split("_")[-1]} triggers, {cfg.trigger_label} class(es) *****\033[0m')

    # warm up with clean data

    criterion = nn.CrossEntropyLoss()
    phases = []
    if 'phase' not in cfg:
        phases = ['1_init', '2_finetune', '3_retrain']
    else:
        if isinstance(cfg.phase, list):
            phases = cfg.phase
        elif isinstance(cfg.phase, str):
            if cfg.phase == 'all':
                phases = ['1_init', '2_finetune', '3_retrain']
            else:
                phases.append(cfg.phase)

    if 'ckpt_suffix' in cfg:
        ckpt_suffix = '_' + cfg.ckpt_suffix
    else:
        ckpt_suffix = ''

    for idx, phase in enumerate(('1_init', '2_finetune', '3_retrain', '2_finetune_mix')):
        print(f"Phase {phase}")
        i = int(phase[0])-1
        if cfg.optimizer[i] == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.lr[i]), weight_decay=float(cfg.wd[i]))
        elif cfg.optimizer[i] == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), float(cfg.lr[i]), momentum=0.9, weight_decay=float(cfg.wd[i]))
    
        if cfg.scheduler[i] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step[i], gamma=cfg.scheduler_gamma[i])
        elif cfg.scheduler[i] == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler_step[i], gamma=cfg.scheduler_gamma[i])
        elif cfg.scheduler[i] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs[i], eta_min=float(cfg.lr_min[i]))

        if phase == '1_init':
            save_path = os.path.join(cfg.save_dir, '1_init', cfg.method, cfg.save_name)
        else:
            save_path = os.path.join(cfg.save_dir, 'ftal', cfg.method, phase, cfg.save_name)
        if phase == '1_init' and phase in phases:
            optimizer, scheduler = get_optims(net, cfg, i)
            if cfg.method == 'rowback':
                evaluator = Evaluator(net, criterion)
                trainer = ROWBACKTrainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
                frozen_layers = [getattr(trainer.net, layer) for layer in cfg.frozen_layers]
                trainer.train(cfg.exp_name, save_path, 10)
                trainer.train_freeze(None, save_path, cfg.epochs[i], train_wm_loader, frozen_layers=frozen_layers, wandb_project=None, eval_pretrain=False, use_wandb=False, save_every=5)
            elif cfg.method == 'ewe':
                trainer = EWETrainer(net, criterion, optimizer, train_loader, test_loader, wm_loader, wm_loader_mix, scheduler=scheduler)
                trainer.train(cfg.exp_name, save_path, cfg.epochs[i])
            else:
                if cfg.method == 'app':
                    evaluator = Evaluator(net, criterion, mark=True)
                    trainer = APPTrainer(net, criterion, optimizer, evaluator, train_wm_loader_app, test_loader_app, wm_loader_app, scheduler=scheduler, batchsize_p=cfg.batchsize_wm, batchsize_c=cfg.batchsize_c, app_eps=cfg.app_eps, app_alpha=cfg.app_alpha)
                elif cfg.method == 'adi':
                    evaluator = Evaluator(net, criterion)
                    trainer = Trainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
                elif cfg.method == 'na':
                    evaluator = Evaluator(net, criterion)
                    trainer = Trainer(net, criterion, optimizer, evaluator, train_loader, test_loader, scheduler=scheduler, use_trigger=False)
                elif cfg.method == 'certified':
                    warmup_epochs = cfg.warmup_epochs if 'warmup_epochs' in cfg else 5
                    evaluator = Evaluator(net, criterion)
                    trainer = CertifiedTrainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True, warmup_epochs=warmup_epochs, avg_times=50)
                trainer.train(cfg.exp_name, save_path, cfg.epochs[i])
        elif phase == '2_finetune' and phase in phases:
            train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True)
            #net.load_state_dict(torch.load(os.path.join(cfg.save_dir, '1_init', cfg.method, cfg.save_name)))
            if 'init_ckpt_path' in cfg and cfg.init_ckpt_path:
                load_path = cfg.init_ckpt_path
            else:
                # 如果配置文件中没有提供 init_ckpt_path，则回退到原始的硬编码路径结构
                load_path = os.path.join(cfg.save_dir, '1_init', cfg.method, cfg.save_name)  
            net.load_state_dict(torch.load(load_path))
            print(f"✅ Model loaded successfully from: {load_path}")  
            optimizer, scheduler = get_optims(net, cfg, i)
            evaluator = Evaluator(net, criterion)
            trainer = Trainer(net, criterion, optimizer, evaluator, ft_loader, test_loader, wm_loader, scheduler=scheduler)
            trainer.train(cfg.exp_name + ckpt_suffix, save_path.replace('.pth', ckpt_suffix + '.pth'), cfg.epochs[i], wandb_project='ftal_finetune', eval_pretrain=True, use_wandb=cfg.log_wandb, eval_robust=cfg.eval_robust)
        elif phase == '3_retrain' and phase in phases:
            train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True)
            net.load_state_dict(torch.load(os.path.join(cfg.save_dir, 'ftal', cfg.method, '2_finetune', cfg.save_name.replace('.pth', ckpt_suffix + '.pth'))))
            optimizer, scheduler = get_optims(net, cfg, i)
            evaluator = Evaluator(net, criterion)
            trainer = Trainer(net, criterion, optimizer, evaluator, train_loader, test_loader, wm_loader, scheduler=scheduler)
            trainer.train(cfg.exp_name + ckpt_suffix, save_path.replace('.pth', ckpt_suffix + '.pth'), cfg.epochs[i], wandb_project='ftal_retrain', eval_pretrain=True, use_wandb=cfg.log_wandb, eval_robust=cfg.eval_robust)
        elif phase == '2_finetune_mix' and phase in phases:
            train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True)
            net.load_state_dict(torch.load(os.path.join(cfg.save_dir, '1_init', cfg.method, cfg.save_name)))
            optimizer, scheduler = get_optims(net, cfg, i)
            evaluator = Evaluator(net, criterion)
            # trainer = Trainer(net, criterion, optimizer, evaluator, train_ft_loader, test_loader, wm_loader, scheduler=scheduler)
            trainer = Trainer(net, criterion, optimizer, evaluator, ft_loader, test_loader, wm_loader, scheduler=scheduler)
            # trainer.train(cfg.exp_name + ckpt_suffix, save_path.replace('.pth', 'mix_' + ckpt_suffix + '.pth'), cfg.epochs[i], wandb_project='ftal_mix', eval_pretrain=True, use_wandb=cfg.log_wandb, eval_robust=cfg.eval_robust)
            trainer.train(cfg.exp_name + ckpt_suffix, save_path.replace('.pth', 'mix_' + ckpt_suffix + '.pth'), cfg.epochs[i], wandb_project='ftal_mix', eval_pretrain=True, use_wandb=cfg.log_wandb, eval_robust=cfg.eval_robust, mix_per_batch=2, mix_loader=train_loader)

    if cfg.visualize:
        print('='*5, 'Generating loss landscape', '='*5)
        cfg.extraction = False
        args = plot_trajectory.plot(cfg)
        args = plot_surface.plot(cfg, args)
        plot_2D.plot(args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='path to config')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
        if cfg.method == 'ewe':
            cfg.ewe = True
            cfg.regenerate = False
    main(cfg)