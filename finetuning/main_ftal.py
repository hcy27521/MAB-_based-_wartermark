import os
import wandb
from argparse import ArgumentParser

import numpy as np
import data_utils
from train_utils import *
from uchida_trainer import UchidaTrainer 
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

    # ----------------------------------------------------------------------
    # 【核弹级修复】强制修正 WandB Entity
    # 无论本地缓存了什么旧团队，这里强制读取当前登录用户，并设置为上传目标
    try:
        api = wandb.Api()
        current_entity = api.default_entity
        os.environ['WANDB_ENTITY'] = current_entity
        print(f"🔒 [Auto-Fix] Enforcing WandB Entity to current user: {current_entity}")
    except Exception as e:
        print(f"⚠️ WandB Entity auto-fix warning: {e}")
        # 如果 API 获取失败，尝试清除环境变量作为兜底
        if 'WANDB_ENTITY' in os.environ:
            del os.environ['WANDB_ENTITY']
    # ----------------------------------------------------------------------

    # data setup
    classes = []
    if cfg.method == 'ewe':
        train_loader, test_loader, wm_loader, wm_loader_mix, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True, ewe=True)
        classes = train_loader.dataset.classes
    elif cfg.method == 'app':
        train_loader_app, test_loader_app, wm_loader_app, train_wm_loader_app, ft_loader_app, train_ft_loader_app = utils.get_data_from_config(cfg, with_mark=True, train_ft_mix=True)
        classes = train_loader_app.dataset.classes
    elif cfg.method in ['na', 'uchida']:
        train_loader, test_loader = utils.get_data_from_config(cfg)
        classes = test_loader.dataset.classes
    else:
        train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True)
        classes = train_loader.dataset.classes

    net = utils.get_model_from_config(cfg, len(classes))
    print(f'\033[1m***** Experiments for WM method {cfg.method}, {cfg.trigger_type.split("_")[-1]} triggers, {cfg.trigger_label} class(es) *****\033[0m')

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

    # 定义默认项目名，避免权限错误
    default_wandb_project = 'finetuning_experiments' 

    for idx, phase in enumerate(('1_init', '2_finetune', '3_retrain', '2_finetune_mix')):
        print(f"Phase {phase}")
        i = int(phase[0])-1
        
        optimizer, scheduler = get_optims(net, cfg, i)
        
        if phase == '1_init':
            save_path = os.path.join(cfg.save_dir, '1_init', cfg.method, cfg.save_name)
        else:
            save_path = os.path.join(cfg.save_dir, 'ftal', cfg.method, phase, cfg.save_name)
            
        if phase == '1_init' and phase in phases:
            
            # === Uchida 独立训练块 ===
            if cfg.method == 'uchida':
                w_lambda = getattr(cfg, 'w_lambda', 0.1)
                embed_dim = getattr(cfg, 'embed_dim', 64)
                target_layer = getattr(cfg, 'target_layer', 'features.0') 
                
                evaluator = Evaluator(net, criterion)
                trainer = UchidaTrainer(
                    net=net, 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    evaluator=evaluator, 
                    train_loader=train_loader, 
                    test_loader=test_loader, 
                    scheduler=scheduler,
                    target_layer_name=target_layer,
                    embed_dim=embed_dim,
                    w_lambda=w_lambda
                )
                trainer.train(cfg.exp_name, save_path, cfg.epochs[i], wandb_project=default_wandb_project, use_wandb=cfg.log_wandb)
                continue 
            # =======================
            
            # 其他方法的逻辑
            elif cfg.method == 'rowback':
                evaluator = Evaluator(net, criterion)
                trainer = ROWBACKTrainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
                frozen_layers = [getattr(trainer.net, layer) for layer in cfg.frozen_layers]
                trainer.train(cfg.exp_name, save_path, 10)
                trainer.train_freeze(None, save_path, cfg.epochs[i], train_wm_loader, frozen_layers=frozen_layers, wandb_project=None, eval_pretrain=False, use_wandb=False, save_every=5)
            elif cfg.method == 'ewe':
                trainer = EWETrainer(net, criterion, optimizer, train_loader, test_loader, wm_loader, wm_loader_mix, scheduler=scheduler)
                trainer.train(cfg.exp_name, save_path, cfg.epochs[i])
            else:
                # 通用 Trainer 初始化
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

                if 'trainer' in locals():
                     # 注意：这里原本可能没有 wandb_project 参数，如果要统一，建议这里不要改动原Trainer调用，
                     # 或者确保原Trainer的train方法支持 **kwargs。
                     # 鉴于之前的报错只针对 Uchida，我们这里保持原样，或者如果你确定 Trainer 支持，可以加上。
                     # 假设原 Trainer 没有修改，保持原样最安全。
                     trainer.train(cfg.exp_name, save_path, cfg.epochs[i]) 
                else:
                    pass

        # Phase 2: Finetune
        elif phase == '2_finetune' and phase in phases:
            if cfg.method in ['uchida', 'na']:
                ft_loader, test_loader = utils.get_data_from_config(cfg)
                wm_loader = None 
            else:
                ft_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True)

            if 'init_ckpt_path' in cfg and cfg.init_ckpt_path:
                load_path = cfg.init_ckpt_path
            else:
                load_path = os.path.join(cfg.save_dir, '1_init', cfg.method, cfg.save_name) 
                            
            net.load_state_dict(torch.load(load_path))
            print(f"✅ Model loaded successfully from: {load_path}") 
            
            optimizer, scheduler = get_optims(net, cfg, i)
            evaluator = Evaluator(net, criterion)

            finetune_save_path = save_path.replace('.pth', ckpt_suffix + '.pth')
            
            if cfg.method == 'uchida':
                # === Uchida Phase 2: 使用 UchidaTrainer，但 w_lambda=0，实现逐 Epoch 监控 WM_Bit_ACC ===
                w_lambda = 0.0 # 纯 Finetune 攻击，不加入水印损失
                embed_dim = getattr(cfg, 'embed_dim', 64)
                target_layer = getattr(cfg, 'target_layer', 'features.0') 
                
                trainer = UchidaTrainer(
                    net=net, 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    evaluator=evaluator, 
                    train_loader=ft_loader, # 使用 Finetune 数据加载器
                    test_loader=test_loader, 
                    scheduler=scheduler,
                    target_layer_name=target_layer,
                    embed_dim=embed_dim,
                    w_lambda=w_lambda # 设为 0，只进行分类任务
                )
                print(f"⚠️ [Uchida Finetune] Using UchidaTrainer with w_lambda=0 to track WM_Bit_ACC per epoch.")
                trainer.train(cfg.exp_name + ckpt_suffix, finetune_save_path, cfg.epochs[i], wandb_project=default_wandb_project, use_wandb=cfg.log_wandb)
            else:
                # === 其他方法 Phase 2: 使用通用 Trainer ===
                trainer = Trainer(net, criterion, optimizer, evaluator, ft_loader, test_loader, wm_loader, scheduler=scheduler)
                trainer.train(cfg.exp_name + ckpt_suffix, finetune_save_path, cfg.epochs[i], wandb_project=default_wandb_project, eval_pretrain=True, use_wandb=cfg.log_wandb, eval_robust=cfg.eval_robust)
            

        # Phase 3: Retrain (已补全)
        elif phase == '3_retrain' and phase in phases:
            train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True)
            # 加载微调后的模型进行重训练
            net.load_state_dict(torch.load(os.path.join(cfg.save_dir, 'ftal', cfg.method, '2_finetune', cfg.save_name.replace('.pth', ckpt_suffix + '.pth'))))
            optimizer, scheduler = get_optims(net, cfg, i)
            evaluator = Evaluator(net, criterion)
            trainer = Trainer(net, criterion, optimizer, evaluator, train_loader, test_loader, wm_loader, scheduler=scheduler)
            # 将项目名称统一为 default_wandb_project，避免权限问题
            trainer.train(cfg.exp_name + ckpt_suffix, save_path.replace('.pth', ckpt_suffix + '.pth'), cfg.epochs[i], wandb_project=default_wandb_project, eval_pretrain=True, use_wandb=cfg.log_wandb, eval_robust=cfg.eval_robust)
        
        # Phase 2 Mix: Finetune Mix (已补全)
        elif phase == '2_finetune_mix' and phase in phases:
            train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = utils.get_data_from_config(cfg, with_mark=False, train_ft_mix=True)
            net.load_state_dict(torch.load(os.path.join(cfg.save_dir, '1_init', cfg.method, cfg.save_name)))
            optimizer, scheduler = get_optims(net, cfg, i)
            evaluator = Evaluator(net, criterion)
            trainer = Trainer(net, criterion, optimizer, evaluator, ft_loader, test_loader, wm_loader, scheduler=scheduler)
            # 将项目名称统一为 default_wandb_project
            trainer.train(cfg.exp_name + ckpt_suffix, save_path.replace('.pth', 'mix_' + ckpt_suffix + '.pth'), cfg.epochs[i], wandb_project=default_wandb_project, eval_pretrain=True, use_wandb=cfg.log_wandb, eval_robust=cfg.eval_robust, mix_per_batch=2, mix_loader=train_loader)


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