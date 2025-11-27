import os
import wandb
from argparse import ArgumentParser
import numpy as np
import data_utils
from train_utils import Trainer, Evaluator # 假设 Trainer, Evaluator, get_optims 在 train_utils 中
import utils
from easydict import EasyDict
import yaml
import torch
from torch import nn
from arch_model_loader import load_arch_model # 导入我们刚创建的模型加载器
import time
from tqdm import tqdm 

# -----------------------------------------------------------------------
# 辅助函数：优化器和调度器 (从您原 main.py 复制过来)
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# 核心训练函数 (简化版，只关注 Phase 1)
# -----------------------------------------------------------------------
def main_train(cfg):
    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_wandb_project = 'arch_watermark_training'
    
    # 1. 数据设置 - 关键修改点
    # 必须接收 utils.get_data_from_config 返回的所有加载器，以避免 unpack 错误
    try:
        data_loaders = utils.get_data_from_config(cfg) 
        
        # 假设最常用的返回是 6 个 (train, test, wm, train_wm, ft, train_ft)
        if len(data_loaders) == 6:
            train_loader, test_loader, wm_loader, train_wm_loader, ft_loader, train_ft_loader = data_loaders
        elif len(data_loaders) == 2:
            # 对应 cfg.method == 'na' 的情况
            train_wm_loader, test_loader = data_loaders
            train_loader = train_wm_loader # 假设 train_wm_loader 可以作为常规 train_loader
            wm_loader = None
        else:
            # 兜底处理，确保至少拿到 train_loader 和 test_loader，并尝试拿 wm_loader
            print(f"⚠️ Warning: utils.get_data_from_config returned {len(data_loaders)} loaders. Assuming the first three are clean_train, clean_test, and wm_test.")
            train_loader = data_loaders[0]
            test_loader = data_loaders[1]
            wm_loader = data_loaders[2] if len(data_loaders) > 2 else None

        classes = test_loader.dataset.classes
        num_classes = len(classes)
    except Exception as e:
        print(f"❌ Error during data loading: {e}. Please ensure data_utils, utils, and all required data are configured.")
        return
    
    # 2. 模型加载 (关键修改点)
    try:
        ModelClass = load_arch_model(cfg.arch_key)
        net = ModelClass().to(device)
        print(f"✅ Successfully loaded architecture model: {cfg.arch_key}")
    except ValueError as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f'\033[1m***** Training Architecture WM: {cfg.arch_key}, Target: {cfg.trigger_label} *****\033[0m')

    criterion = nn.CrossEntropyLoss()
    
    # 我们只关注 '1_init' (初始训练)
    phase_idx = 0 
    
    # 获取优化器和调度器 (使用配置中的第一组参数)
    optimizer, scheduler = get_optims(net, cfg, phase_idx)
    
    # 定义保存路径
    # 命名规则：[arch_key]_[target_label]_best.pth
    save_name = f"{cfg.arch_key}_{cfg.trigger_label}_best.pth"
    save_dir = os.path.join(cfg.save_dir, 'arch_weights')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    # 3. Trainer 初始化并训练
    evaluator = Evaluator(net, criterion)
    
    # 注意：我们使用普通的 Trainer，因为水印逻辑已在模型 forward() 中实现
    # 如果您的 Trainer 需要额外的 wm_loader，请根据实际情况调整 data_utils.py 或此处的调用
    trainer = Trainer(
        net=net, 
        criterion=criterion, 
        optimizer=optimizer, 
        evaluator=evaluator, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        scheduler=scheduler
        # 假设您的 Trainer 支持 wm_loader=None
    )
    
    print(f"🚀 Starting training for {cfg.epochs[phase_idx]} epochs...")
    # 调用训练，并记录到 WANDB
    trainer.train(
        exp_name=f"TRAIN_{cfg.arch_key}", 
        save_path=save_path, 
        epochs=cfg.epochs[phase_idx], 
        wandb_project=default_wandb_project, 
        use_wandb=cfg.log_wandb
    )
    
    print(f"✅ Training complete. Weights saved to: {save_path}")
    print(f"WGT_PATH for evaluation is: {save_path}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    args = parser.parse_args()
    
    try:
        with open(args.config) as f:
            cfg = EasyDict(yaml.safe_load(f))
            # 确保 epochs 是一个列表，以匹配 get_optims 的逻辑
            if not isinstance(cfg.epochs, list):
                cfg.epochs = [cfg.epochs] 
            if not isinstance(cfg.lr, list):
                cfg.lr = [cfg.lr]
            # ... 对其他相位相关的配置进行类似处理，以适配 get_optims
            
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit()
        
    main_train(cfg)