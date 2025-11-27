import torch
import torch.nn as nn
import numpy as np
import os 
import wandb
# 导入 train_utils 中的 init_wandb 函数，用于初始化 WandB
from train_utils import init_wandb 

class UchidaTrainer:
    def __init__(self, net, criterion, optimizer, evaluator, train_loader, test_loader, scheduler=None, 
                 target_layer_name='features.0', embed_dim=64, w_lambda=0.1, device='cuda'):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.device = device
        self.w_lambda = w_lambda
        self.embed_dim = embed_dim
        
        # --- 目标层参数定位逻辑 --- (保持不变)
        self.target_param = None
        for name, param in self.net.named_parameters():
            if name.startswith(target_layer_name) and 'weight' in name:
                self.target_param = param
                break
        
        if self.target_param is None:
            for name, param in self.net.named_parameters():
                if name == target_layer_name:
                    self.target_param = param
                    break
        
        if self.target_param is None:
            if target_layer_name == 'features.0':
                for name, param in self.net.named_parameters():
                    if 'features.0.weight' in name:
                        self.target_param = param
                        break

        if self.target_param is None:
            print(f"Available layers:")
            for n, _ in self.net.named_parameters():
                print(n)
            raise ValueError(f"Critical Error: Layer '{target_layer_name}' not found in model.")
            
        print(f"[*] Uchida Watermark targeting layer: {target_layer_name}, shape: {self.target_param.shape}")

        flat_dim = np.prod(self.target_param.shape)
        rng = torch.Generator()
        rng.manual_seed(42) 
        self.X = torch.randn(self.embed_dim, flat_dim, generator=rng).to(self.device)
        self.b = torch.bernoulli(torch.full((self.embed_dim,), 0.5)).to(self.device) 

    def get_watermark_loss(self):
        w_flat = self.target_param.view(-1)
        projection = torch.matmul(self.X, w_flat)
        loss_w = nn.BCEWithLogitsLoss()(projection, self.b)
        return loss_w

    def get_watermark_acc(self):
        with torch.no_grad():
            w_flat = self.target_param.view(-1)
            projection = torch.matmul(self.X, w_flat)
            preds = (torch.sigmoid(projection) > 0.5).float()
            correct_bits = (preds == self.b).float().sum()
            acc = correct_bits / self.embed_dim
        return acc.item()

    # 【修复点 1：添加 wandb 相关参数，并设置默认值】
    def train(self, exp_name, save_path, epochs, wandb_project=None, use_wandb=False, eval_pretrain=False, eval_robust=False):
        
        # 【修复点 2：在训练开始时初始化 WandB】
        config = {'w_lambda': self.w_lambda, 'embed_dim': self.embed_dim}
        init_wandb(wandb_project, exp_name, config, use_wandb)

        best_acc = 0
        print(f"Start training Uchida Watermark for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.net.train()
            loss_meter = 0.0
            w_loss_meter = 0.0
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.net(inputs)
                task_loss = self.criterion(outputs, targets)
                w_loss = self.get_watermark_loss()
                
                loss = task_loss + self.w_lambda * w_loss
                
                loss.backward()
                self.optimizer.step()
                
                loss_meter += loss.item()
                w_loss_meter += w_loss.item()

            if self.scheduler:
                self.scheduler.step()

            clean_acc_raw = self.evaluator.eval(self.test_loader)
            
            # 智能解析 clean_acc (保持不变)
            clean_acc = 0.0
            if isinstance(clean_acc_raw, dict):
                if 'acc' in clean_acc_raw:
                    clean_acc = clean_acc_raw['acc']
                elif 'top1' in clean_acc_raw:
                    clean_acc = clean_acc_raw['top1']
                elif 'accuracy' in clean_acc_raw:
                    clean_acc = clean_acc_raw['accuracy']
                else:
                    if clean_acc_raw:
                        clean_acc = list(clean_acc_raw.values())[0]
            elif isinstance(clean_acc_raw, (float, int)):
                clean_acc = clean_acc_raw
            elif torch.is_tensor(clean_acc_raw):
                clean_acc = clean_acc_raw.item()
            
            wm_acc = self.get_watermark_acc()
            
            log_dict = {
                "Loss/Train Loss": loss_meter/len(self.train_loader),
                "Loss/Watermark Loss": w_loss_meter/len(self.train_loader),
                "Acc/Clean Test Acc": clean_acc,
                "Acc/WM Bit Acc": wm_acc * 100,
                "Misc/LR": self.optimizer.param_groups[0]['lr'],
                "Epoch": epoch + 1
            }
            
            # 【修复点 3：使用 WandB 记录日志】
            if use_wandb:
                wandb.log(log_dict)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_meter/len(self.train_loader):.4f} | "
                  f"W_Loss: {w_loss_meter/len(self.train_loader):.4f} | "
                  f"ACC: {clean_acc:.2f}% | WM_Bit_ACC: {wm_acc*100:.2f}%")

            if clean_acc > best_acc:
                best_acc = clean_acc
                # 确保父目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.net.state_dict(), save_path)