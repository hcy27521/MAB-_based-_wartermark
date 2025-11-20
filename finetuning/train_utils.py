import torch
import os
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import wandb
import math
from copy import deepcopy
from app import AdversarialParameterPerturbation
from utils import ListLoader
from models import ResNetCBN
import pandas as pd
from wrt.defenses.watermark.jia import Jia
from wrt.classifiers import PyTorchClassifier

class Trainer:
    def __init__(self, net:nn.Module, criterion, optimizer, evaluator, train_loader, test_loader, wm_loader=None, scheduler=None, use_trigger=False) -> None:
        self.net = net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.wm_loader = wm_loader
        self.use_trigger = use_trigger
        self.scheduler = scheduler

    def train_epoch(self, loader=None, mix_per_batch=False, mix_loader=None):
        if loader is None:
            loader = self.train_loader
        self.net.train()
        data_size = 0
        # if self.wm_loader and self.use_trigger:
        #     wm_inputs, wm_targets = [], []
        #     for inputs, targets in self.wm_loader:
        #         # inputs, targets = inputs.to(self.device), targets.to(self.device)
        #         wm_inputs.append(inputs)
        #         wm_targets.append(targets)
        #     wm_idx = np.random.randint(len(wm_inputs))
        # train_inputs, train_targets = [], []
        # if mix_per_batch:
        #     for inputs, targets in self.train_loader:
        #         inputs, targets = inputs.to(self.device), targets.to(self.device)
        #         train_inputs.append(inputs)
        #         train_targets.append(targets)
        # train_idx = np.random.randint(len(train_inputs))
        running_loss = 0.0
        running_corrects = 0
        loader_it = iter(loader)
        if mix_loader:
            mix_loader_it = iter(mix_loader)
        progress_bar = tqdm(range(len(loader)))
        for batch_idx in progress_bar:
            batch = next(loader_it)
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            # if self.wm_loader and self.use_trigger and mix_per_batch:
            #     inputs = torch.cat([inputs, wm_inputs[(wm_idx + batch_idx)%len(wm_inputs)].to(self.device)], dim=0)
            #     targets = torch.cat([targets, wm_targets[(wm_idx + batch_idx)%len(wm_targets)].to(self.device)], dim=0)
            if type(mix_per_batch) == int and mix_loader:
                if (batch_idx+1) % mix_per_batch == 0:
                    try:
                        mix_inputs, mix_targets = next(mix_loader_it)
                    except StopIteration:
                        mix_loader_it = iter(mix_loader)
                        mix_inputs, mix_targets = next(mix_loader_it)
                    mix_inputs, mix_targets = mix_inputs.to(self.device), mix_targets.to(self.device)
                    self.optimizer.zero_grad()
                    mix_outputs = self.net(mix_inputs)
                    _, preds = torch.max(mix_outputs, 1)
                    loss = self.criterion(mix_outputs, mix_targets)
                    loss.backward()
                    self.optimizer.step()
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets.data).item()
            data_size += inputs.size(0)
            progress_bar.set_description(f"Loss {running_loss/data_size:.3f}, Acc {running_corrects/data_size*100:.2f}")
        metrics = {
            "loss": running_loss / data_size,
            "accuracy": running_corrects / data_size * 100
        }
        if self.scheduler:
            self.scheduler.step()
            print(f"Last LR={self.scheduler.get_last_lr()}")
        return metrics

    # def train(trainer, evaluator, test_loader, epochs, exp_name, savename, frozen_layers=[], wm_loader=None, project="watermark", pretrain_eval=False, mix_per_batch=False, eval_robust=False, use_wandb=False, save_epoch=5, calc_diff=True, orig_model=None):
    def train(self, exp_name, save_name, epochs, loader=None, wandb_project=None, eval_robust=False, eval_pretrain=False, use_wandb=False, save_every=5, mix_loader=None, mix_per_batch=False):
        metrics = {}
        print('Start Training...', flush=True)
        config = {
            'lr': self.optimizer.param_groups[0]['lr'],
            'epoch': epochs
        }
        init_wandb(wandb_project, exp_name, config, use_wandb)
        begin_epoch = time.time()
        best_val_acc = 0.0
        # best_wm_acc = 0.0
        # writer = SummaryWriter(log_dir=os.path.join(logdir, logcmt))
        if self.wm_loader is not None:
            metrics = {
                'epoch': [],
                'train_acc': [],
                'train_loss': [],
                'val_acc': [],
                'val_loss': [],
                'wm_acc': [],
                'wm_loss': []
            }
        else:
            metrics = {
                'epoch': [],
                'train_acc': [],
                'train_loss': [],
                'val_acc': [],
                'val_loss': [],
            }
        dirname = save_name.rsplit('/', 1)[0]
        if len(save_name.rsplit('/', 1)) > 1:
            os.makedirs(dirname, exist_ok=True)
        name_split = save_name.rsplit('.', 1)
        torch.save(self.net.state_dict(), name_split[0] + f'_0.'+ name_split[1])
        if eval_pretrain:
            val_metrics = self.evaluator.eval(self.test_loader)
            log_wandb({'val/acc': val_metrics['accuracy'], 'val/loss': val_metrics['loss']}, step=0, log=use_wandb)
            metrics['epoch'].append(0)
            metrics['val_acc'] = [val_metrics['accuracy']]
            metrics['val_loss'] = [val_metrics['loss']]
            if self.wm_loader is not None:
                trigger_metrics = self.evaluator.eval(self.wm_loader)
                log_wandb({'trigger/acc': trigger_metrics['accuracy'], 'trigger/loss': trigger_metrics['loss']}, step=0, log=use_wandb)
                metrics['wm_acc'] = [trigger_metrics['accuracy']]
                metrics['wm_loss'] = [trigger_metrics['loss']]
                if eval_robust:
                    wm_acc_avg, wm_acc_median = self.evaluator.eval_robust(self.wm_loader)
                    log_wandb({'trigger/avg_acc': wm_acc_avg, 'trigger/median_acc': wm_acc_median}, step=0, log=use_wandb)
                    metrics['trigger_avg_acc'] = [wm_acc_avg]
                    metrics['trigger_median_acc'] = [wm_acc_median]
        # for layer in frozen_layers:
        #     freeze_layer(layer)
        self.epoch = 0
        for epoch in range(1, epochs+1):
            # if frozen_layers:
            #     unfrozen_layer = frozen_layers[epoch % len(frozen_layers)]
            #     unfreeze_layer(unfrozen_layer)
            train_metrics = self.train_epoch(loader, mix_per_batch, mix_loader)
            val_metrics = self.evaluator.eval(self.test_loader)
            metrics['train_acc'].append(train_metrics['accuracy'])
            metrics['train_loss'].append(train_metrics['loss'])
            metrics['val_acc'].append(val_metrics['accuracy'])
            metrics['val_loss'].append(val_metrics['loss'])
            log_wandb({'train/acc': train_metrics['accuracy'], 'train/loss': train_metrics['loss'],
                    'val/acc': val_metrics['accuracy'], 'val/loss': val_metrics['loss']}, step=epoch, log=use_wandb)
            metrics['epoch'].append(epoch)
            if self.wm_loader is not None:
                trigger_metrics = self.evaluator.eval(self.wm_loader)
                metrics['wm_acc'].append(trigger_metrics['accuracy'])
                metrics['wm_loss'].append(trigger_metrics['loss'])
                if eval_robust:
                    wm_acc_avg, wm_acc_median = self.evaluator.eval_robust(self.wm_loader)
                    log_wandb({'trigger/avg_acc': wm_acc_avg, 'trigger/median_acc': wm_acc_median}, step=epoch, log=use_wandb)
                    metrics['trigger_avg_acc'].append(wm_acc_avg)
                    metrics['trigger_median_acc'].append(wm_acc_median)
                log_wandb({'trigger/acc': trigger_metrics['accuracy'], 'trigger/loss': trigger_metrics['loss']}, step=epoch, log=use_wandb)
                # writer.add_scalar("Loss/trigger", trigger_metrics['loss'], epoch)
                # writer.add_scalar("Accuracy/trigger", trigger_metrics['accuracy'], epoch)
                if not eval_robust:
                    print(
                        f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                        f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                        f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}"
                        f"| Trigger Loss {trigger_metrics['loss']:.3f} | Trigger Acc {trigger_metrics['accuracy']:.2f}",
                        flush=True)
                else:
                    print(
                        f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                        f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                        f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}"
                        f"| Trigger Loss {trigger_metrics['loss']:.3f} | Trigger Acc {trigger_metrics['accuracy']:.2f} | Trigger Acc Avg {wm_acc_avg:.2f} | Trigger Acc Median {wm_acc_median:.2f}",
                        flush=True)
                # if len(frozen_layers):
                #     freeze_layer(unfrozen_layer)
            else:
                print(
                    f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                    f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                    f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}",
                    flush=True)
            torch.save(self.net.state_dict(), save_name)
            if (epoch == 1) or (epoch % save_every == 0):
                torch.save(self.net.state_dict(), name_split[0] + f'_{epoch}.' + name_split[1])
        print()
        # metrics_df = pd.DataFrame(metrics)
        # metrics_df.to_csv(exp_name + '.csv', index=False)
        finish_wandb(log=use_wandb)
        return metrics

class APPTrainer(Trainer): 
    def __init__(self, *args, batchsize_p, batchsize_c, app_eps, app_alpha, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app_eps = float(app_eps)
        self.app_alpha = float(app_alpha)
        self.cbs = batchsize_c
        self.pbs = batchsize_p

    def train_epoch(self, loader=None, mix_per_batch=False, mix_loader=None):
        if loader is None:
            loader = self.train_loader
        self.net.train()
        correct, loss_, n = 0., 0., 0.    
        pcorrect, ploss_, pn = 0., 0., 0.            
        app_norm = 'rl2'
        # poison_listloader = ListLoader(self.dataset.get_poison_components_train_loader(batch_size=p_batch_size))
        app = AdversarialParameterPerturbation(app_norm, eps=self.app_eps)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            loss_fn = F.cross_entropy
        else:
            raise NotImplementedError("Not Implemented for Loss", type(self.criterion))
        train_iter = iter(self.train_loader)
        trigger_iter = iter(self.wm_loader)
        for b_idx in tqdm(range(len(self.train_loader))):
            cimgs, cy, cmask = next(train_iter)
            self.optimizer.zero_grad()   
            # prepare data
            try:
                batch = next(trigger_iter)
            except StopIteration:
                trigger_iter = iter(self.wm_loader)
                batch = next(trigger_iter)
            pimgs, py, pmask = batch
            cimgs, cy, cmask  = cimgs.cuda(), cy.cuda(), cmask.cuda()
            pimgs, py, pmask  = pimgs.cuda(), py.cuda(), pmask.cuda()            
            mixed_imgs, mixed_ys, mixed_mask = torch.cat([pimgs, cimgs[:self.cbs]]), torch.cat([py, cy[:self.cbs]]), torch.cat([pmask, cmask[:self.cbs]])
            
            # calculate perturabation using mixed data
            freeze_bn(self.net, True)
            if isinstance(self.net, ResNetCBN):
                loss_fn(self.net(mixed_imgs, mixed_mask), mixed_ys, reduction='none')[:self.pbs].mean().backward()
            else:
                loss_fn(self.net(mixed_imgs), mixed_ys, reduction='none')[:self.pbs].mean().backward()
            # loss_fn(model(mixed_imgs, mixed_mask), mixed_ys, reduction='none')[mixed_mask==1].mean().backward()
            perturbation = app.calc_perturbation(self.net)
            # calculate watermark grad on perturbed model
            self.net.zero_grad()
            app.perturb(self.net, perturbation)
            # poutputs = self.net(mixed_imgs, mixed_mask)[:pbs]
            if isinstance(self.net, ResNetCBN):
                poutputs = self.net(mixed_imgs, mixed_mask)[:self.pbs]
            else:
                poutputs = self.net(mixed_imgs)[:self.pbs]
            # poutputs = model(mixed_imgs, mixed_mask)[mixed_mask==1]
            ploss = self.criterion(poutputs, py)
            (self.app_alpha*ploss).backward()
            app.restore(self.net, perturbation)
            freeze_bn(self.net, False)
            # calculate grad on unperturbed model
            coutputs = self.net(cimgs)
            closs = self.criterion(coutputs, cy)
            closs.backward()
            self.optimizer.step()
            # collect results
            pcorrect += (poutputs.max(1)[1]==py).sum().item()
            ploss_ += ploss.item() * len(py)
            pn += len(py)
            
            correct += (coutputs.max(1)[1]==cy).sum().item()            
            loss_ +=  closs.item()* cy.shape[0]
            n += cy.shape[0]
        acc = correct / n *100.0
        loss_ = loss_ / n
        if pn:
            asr = pcorrect / pn * 100.0
            ploss_ = ploss_ /pn
        else:
            asr = ploss_ = 0.
        metrics = {
            "loss": loss_,
            "accuracy": acc
        }
        if self.scheduler:
            print(f"Last LR={self.scheduler.get_last_lr()}")
            self.scheduler.step()
        return metrics
        # return [acc, loss_, asr, ploss_]

    def id_dir(self):
        args = self.args
        return '_'.join([args.method, "a%1.2e"%args.alpha, args.app_norm, 'eps%1.2e'%args.app_eps, 'pbs%d'%args.p_batch_size, 'bbs%d'%args.b_batch_size])

class CertifiedTrainer(Trainer):
    def __init__(self, *args, robust_noise_step=0.05, robust_noise=1.0, avg_times=100, warmup_epochs=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.robust_noise_step = robust_noise_step
        self.robust_noise = robust_noise
        self.avg_times = avg_times
        self.warmup_epochs = warmup_epochs

    # def train_robust(net, wm_loader, optimizer, criterion, robust_noise, robust_noise_step, avgtimes):
    def train_robust(self):
        self.net.train()
        wm_train_accuracy = 0.0
        loader_it = iter(self.wm_loader)
        running_corrects = 0
        data_size = 0
        for i in tqdm(range(len(self.wm_loader)), desc="Train robust"):
            data = next(loader_it)
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            times = int(self.robust_noise / self.robust_noise_step) + 1
            in_times = self.avg_times
            for j in range(times):
                self.optimizer.zero_grad()
                for k in range(in_times):
                    Noise = {}
                    # Add noise
                    for name, param in self.net.named_parameters():
                        gaussian = torch.randn_like(param.data) * 1
                        Noise[name] = self.robust_noise_step * j * gaussian
                        param.data = param.data + Noise[name]

                    # get the inputs
                    outputs = self.net(inputs)
                    class_loss = self.criterion(outputs, labels)
                    loss = class_loss / (times * in_times)
                    loss.backward()

                    # remove the noise
                    for name, param in self.net.named_parameters():
                        param.data = param.data - Noise[name]

                self.optimizer.step()
            max_vals, max_indices = torch.max(outputs, 1)
            running_corrects += (max_indices == labels).sum().item()
            data_size += inputs.size(0)

        wm_train_accuracy = running_corrects / data_size * 100
        return wm_train_accuracy


    # def train_normal(net, loader, optimizer, criterion):
    def train_normal(self):
        self.net.train()
        train_accuracy = 0.0
        train_iter = iter(self.train_loader)
        data_size = 0
        running_corrects = 0
        running_loss = 0.0
        for i in tqdm(range(len(self.train_loader)), desc="Train normal"):
            # get the inputs
            data = next(train_iter)
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            max_vals, max_indices = torch.max(outputs, 1)
            running_corrects += (max_indices == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)
            data_size += inputs.size(0)

        train_accuracy = running_corrects / data_size * 100
        train_loss = running_loss / data_size
        return train_accuracy, train_loss


    # def train_certified(net, train_watermark_loader, wm_loader, test_loader, optimizer, scheduler, savename, epochs=50, warmup_epochs=5, robust_noise_step=0.05, robust_noise=1.0, avgtimes=100, train_loader=None, save_epoch=5):
    def train(self, exp_name, save_name, epochs, save_every=5):
        self.net.cuda()
        dirname = save_name.rsplit('/', 1)[0]
        if len(save_name.rsplit('/', 1)) > 1:
            os.makedirs(dirname, exist_ok=True)
        name_split = save_name.rsplit('.', 1)
        torch.save(self.net.state_dict(), name_split[0] + f'_0.'+ name_split[1])
        for epoch in range(1, epochs+1):
            # certified robustness starts after a warm start
            wm_train_acc = 0.0
            if epoch > self.warmup_epochs:
                # wm_train_accuracy = self.train_robust(net, wm_loader, optimizer, criterion, robust_noise, robust_noise_step, avgtimes)
                wm_train_acc = self.train_robust()
            # train_accuracy, train_loss = self.train_normal(net, train_watermark_loader, optimizer, criterion)
            train_acc, train_loss = self.train_normal()
            #################################################################################################3
            # EVAL
            ##############################3

            wm_metrics = self.evaluator.eval(self.wm_loader)

            # A new classifier g
            # times = self.avg_times
            # self.net.eval()
            # wm_train_accuracy_avg = 0.0
            # for j in range(times):

            #     Noise = {}
            #     # Add noise
            #     for name, param in self.net.named_parameters():
            #         gaussian = torch.randn_like(param.data)
            #         Noise[name] = self.robust_noise * gaussian
            #         param.data = param.data + Noise[name]

            #     wm_train_accuracy_local = self.evaluator.eval(self.wm_loader)['accuracy']
            #     wm_train_accuracy_avg += wm_train_accuracy_local

            #     # remove the noise
            #     for name, param in self.net.named_parameters():
            #         param.data = param.data - Noise[name]

            # wm_train_accuracy_avg /= times

            val_acc = self.evaluator.eval(self.test_loader)['accuracy']
            self.scheduler.step()

            print("Epoch " + str(epoch))
            print("Train")
            print(f"Train acc {train_acc:.2f} | Train loss {train_loss:.3f} | WM train acc {wm_train_acc:.2f}")
            print("Tests")
            # print(f"WM acc {wm_metrics['accuracy']:.2f} | WM train avg acc {wm_train_accuracy_avg:.2f} | Test acc {val_acc:.2f}")
            print(f"WM acc {wm_metrics['accuracy']:.2f} | Test acc {val_acc:.2f}")


            torch.save(self.net.state_dict(), save_name)
            if (epoch == 1) or (epoch % save_every == 0):
                torch.save(self.net.state_dict(), name_split[0] + f'_{epoch}.'+ name_split[1])

class ROWBACKTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_freeze(self, exp_name, save_name, epochs, loader=None, frozen_layers=None, wandb_project=None, eval_robust=False, eval_pretrain=False, use_wandb=False, save_every=5):
        print('Start Training...', flush=True)
        config = {
            'lr': self.optimizer.param_groups[0]['lr'],
            'epoch': epochs
        }
        init_wandb(wandb_project, exp_name, config, use_wandb)
        begin_epoch = time.time()
        best_val_acc = 0.0
        metrics = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': [],
            'wm_acc': [],
            'wm_loss': []
        }
        dirname = save_name.rsplit('/', 1)[0]
        if len(save_name.rsplit('/', 1)) > 1:
            os.makedirs(dirname, exist_ok=True)
        name_split = save_name.rsplit('.', 1)
        torch.save(self.net.state_dict(), name_split[0] + f'_0.'+ name_split[1])
        if eval_pretrain:
            val_metrics = self.evaluator.eval(self.test_loader)
            log_wandb({'val/acc': val_metrics['accuracy'], 'val/loss': val_metrics['loss']}, step=0, log=use_wandb)
            if self.wm_loader is not None:
                trigger_metrics = self.evaluator.eval(self.wm_loader)
                log_wandb({'trigger/acc': trigger_metrics['accuracy'], 'trigger/loss': trigger_metrics['loss']}, step=0, log=use_wandb)
                if eval_robust:
                    wm_acc_avg, wm_acc_median = self.evaluator.eval_robust(self.wm_loader)
                    log_wandb({'trigger/avg_acc': wm_acc_avg, 'trigger/median_acc': wm_acc_median}, step=0, log=use_wandb)
        for epoch in range(1, epochs+1):
            if frozen_layers:
                for layer in frozen_layers:
                    freeze_layer(layer)
                unfrozen_layer = frozen_layers[epoch % len(frozen_layers)]
                unfreeze_layer(unfrozen_layer)
            train_metrics = self.train_epoch(loader)
            val_metrics = self.evaluator.eval(self.test_loader)
            metrics['train_acc'].append(train_metrics['accuracy'])
            metrics['train_loss'].append(train_metrics['loss'])
            metrics['val_acc'].append(val_metrics['accuracy'])
            metrics['val_loss'].append(val_metrics['loss'])
            log_wandb({'train/acc': train_metrics['accuracy'], 'train/loss': train_metrics['loss'],
                    'val/acc': val_metrics['accuracy'], 'val/loss': val_metrics['loss']}, step=epoch, log=use_wandb)
            if self.wm_loader is not None:
                trigger_metrics = self.evaluator.eval(self.wm_loader)
                metrics['wm_acc'].append(trigger_metrics['accuracy'])
                metrics['wm_loss'].append(trigger_metrics['loss'])
                if eval_robust:
                    wm_acc_avg, wm_acc_median = self.evaluator.eval_robust(self.wm_loader)
                    log_wandb({'trigger/avg_acc': wm_acc_avg, 'trigger/median_acc': wm_acc_median}, step=epoch, log=use_wandb)
                log_wandb({'trigger/acc': trigger_metrics['accuracy'], 'trigger/loss': trigger_metrics['loss']}, step=epoch, log=use_wandb)
                if not eval_robust:
                    print(
                        f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                        f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                        f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}"
                        f"| Trigger Loss {trigger_metrics['loss']:.3f} | Trigger Acc {trigger_metrics['accuracy']:.2f}",
                        flush=True)
                else:
                    print(
                        f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                        f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                        f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}"
                        f"| Trigger Loss {trigger_metrics['loss']:.3f} | Trigger Acc {trigger_metrics['accuracy']:.2f} | Trigger Acc Avg {wm_acc_avg:.2f} | Trigger Acc Median {wm_acc_median:.2f}",
                        flush=True)
            else:
                print(
                    f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                    f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                    f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}",
                    flush=True)
            for layer in frozen_layers:
                unfreeze_layer(layer)
            torch.save(self.net.state_dict(), save_name)
            if (epoch == 1) or (epoch % save_every == 0):
                torch.save(self.net.state_dict(), name_split[0] + f'_{epoch}.' + name_split[1])
        print()
        finish_wandb(log=use_wandb)
        return metrics

class MABTrainer(Trainer):
    """
    用于MAB (Model Architecture Backdoor) 方法的训练器。
    该方法的主要后门逻辑已嵌入到模型架构（如EvilVGG11）中，
    因此继承基础的Trainer类并使用标准的训练流程即可。
    """
    def __init__(self, *args, **kwargs) -> None:
        # 直接调用父类的初始化方法，继承所有的属性和基础训练逻辑
        super().__init__(*args, **kwargs)
        
        # 可以在这里添加 MAB 特有的初始化逻辑（如果需要），例如：
        # print("MAB Trainer initialized. Using standard training loop.")
        
    # 如果MAB有特殊的训练周期逻辑，可以重写 train_epoch 或 train 方法。
    # 如果使用标准的DataLoader（包含Clean Data + Watermark Data），则无需重写。
    # 
    # def train_epoch(self, loader=None, mix_per_batch=False, mix_loader=None):
    #     # ... (如果需要特殊的训练逻辑，在这里重写)
    #     return super().train_epoch(loader, mix_per_batch, mix_loader)

class EWETrainer:
    def __init__(self, net:nn.Module, criterion, optimizer, train_loader, test_loader, wm_loader, wm_loader_mix, scheduler=None, use_trigger=False) -> None:
        self.net = net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.wm_loader = wm_loader
        self.wm_loader_mix = wm_loader_mix
        self.use_trigger = use_trigger
        self.scheduler = scheduler

    def train(self, exp_name, save_path, epochs):
        self.setup_ewe()
        self.embed(save_path, epochs)
        
    def setup_ewe(self):
        classifier = PyTorchClassifier(self.net, self.criterion, self.optimizer, (3,32,32), 10)
        self.ewe = Jia(classifier, self.train_loader, self.test_loader, self.wm_loader, self.wm_loader_mix, snnl_weight=64, num_classes=10, rate=10)
        # self.ewe.classifier.model.load_state_dict(torch.load('checkpoints/1_init/ewe/ewe_noise_resnet_cifar10.pth'))
        
    def embed(self, save_path, epochs):
        self.ewe.embed(self.train_loader, self.test_loader, self.wm_loader, self.wm_loader_mix, save_name=save_path, source_class=0, target_class=0, epochs=epochs, keylength=100, save_every=5)


class ModelExtractor:
    def __init__(self, net, criterion, optimizer, evaluator, train_loader, test_loader, scheduler=None):
        self.net
        self.extracted_net = extracted_net
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_epoch(self, loader=None):
        if loader is None:
            loader = self.train_loader
        data_size = 0
        running_loss = 0.0
        running_corrects = 0
        loader_it = iter(loader)
        self.extracted_net.train()
        progress_bar = tqdm(range(len(loader)))
        for batch_idx in progress_bar:
            batch = next(loader_it)
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            with torch.no_grad():
                outputs_v = self.victim_net(inputs)
                _, preds_v = torch.max(outputs_v, 1)
            self.optimizer.zero_grad()
            outputs_t = self.extracted_net(inputs)
            _, preds_t = torch.max(outputs_t, 1)
            loss = self.criterion(outputs_t, preds_v)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds_t == preds_v).item()
            data_size += inputs.size(0)
            progress_bar.set_description(f"Loss {running_loss/data_size:.3f}, Acc {running_corrects/data_size*100:.2f}")
        metrics = {
            "loss": running_loss / data_size,
            "accuracy": running_corrects / data_size * 100
        }
        if self.scheduler:
            self.scheduler.step()
            print(f"Last LR={self.scheduler.get_last_lr()}")
        return metrics

    def train(self, save_name, epochs, loader=None, wandb_project=None, eval_robust=False, eval_pretrain=False, use_wandb=False, save_every=5, mix_per_batch=False):
        print('Start Training...', flush=True)
        begin_epoch = time.time()
        metrics = {
            'epoch': [],
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': [],
        }
        # dirname = save_name.rsplit('/', 1)[0]
        # if len(save_name.rsplit('/', 1)) > 1:
        #     os.makedirs(dirname, exist_ok=True)
        name_split = save_name.rsplit('.', 1)
        self.victim_net.to(self.device).eval()
        self.extracted_net.to(self.device).train()
        self.epoch = 0
        for epoch in range(1, epochs+1):
            train_metrics = self.train_epoch(loader)
            val_metrics = self.evaluator.eval(self.test_loader)
            metrics['train_acc'].append(train_metrics['accuracy'])
            metrics['train_loss'].append(train_metrics['loss'])
            metrics['val_acc'].append(val_metrics['accuracy'])
            metrics['val_loss'].append(val_metrics['loss'])
            # log_wandb({'train/acc': train_metrics['accuracy'], 'train/loss': train_metrics['loss'],
            #         'val/acc': val_metrics['accuracy'], 'val/loss': val_metrics['loss']}, step=epoch, log=use_wandb)
            metrics['epoch'].append(epoch)
            print(
                f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}",
                flush=True)
            # torch.save(self.net.state_dict(), save_name)
            # if (epoch == 1) or (epoch % save_every == 0):
            #     torch.save(self.net.state_dict(), name_split[0] + f'_{epoch}.' + name_split[1])
        # metrics_df = pd.DataFrame(metrics)
        # metrics_df.to_csv(exp_name + '.csv', index=False)
        # finish_wandb(log=use_wandb)
        return metrics

class Evaluator:
    def __init__(self, net, criterion, mark=False) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net
        self.net.to(self.device)
        self.criterion = criterion
        self.mark = mark

    def eval(self, dataloader):
        self.net.eval()
        data_size = 0
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets.data).item()
                data_size += inputs.size(0)
        metrics = {
            "loss": running_loss / data_size,
            "accuracy": running_corrects / data_size * 100
        }
        return metrics

    def eval_robust(self, wm_loader, times=50, robust_noise=1.0):
        Array = []
        wm_train_accuracy_avg = 0.0
        for j in range(times):

            Noise = {}
            # Add noise
            for name, param in self.net.named_parameters():
                gaussian = torch.randn_like(param.data)
                Noise[name] = robust_noise * gaussian
                param.data = param.data + Noise[name]

            wm_train_accuracy = 0.0
            for i, data in enumerate(wm_loader, 0):
                inputs, labels, _ = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.net(inputs)
                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                wm_train_accuracy += 100 * correct

            wm_train_accuracy /= len(wm_loader)
            wm_train_accuracy_avg += wm_train_accuracy
            Array.append(wm_train_accuracy)

            # remove the noise
            for name, param in self.net.named_parameters():
                param.data = param.data - Noise[name]

        wm_train_accuracy_avg /= times
        Array.sort()
        wm_median = Array[int(len(Array) / 2)]

        return wm_train_accuracy_avg, wm_median

def freeze_bn(model, freeze=False):
    train = not freeze
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = train
            m.weight.requires_grad_(train)
            m.bias.requires_grad_(train)

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
        
def unfreeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = True



def train_robust(net, wm_loader, optimizer, robust_noise, robust_noise_step, avgtimes=100):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    wm_train_accuracy = 0.0
    wm_it = iter(wm_loader)
    wm_size = 0
    for i in tqdm(range(len(wm_loader)), desc='Robust training'):
        data = next(wm_it)
        times = int(robust_noise / robust_noise_step) + 1
        in_times = avgtimes
        for j in range(times):
            optimizer.zero_grad()
            for k in range(in_times):
                Noise = {}
                # Add noise
                for name, param in net.named_parameters():
                    gaussian = torch.randn_like(param.data) * 1
                    Noise[name] = robust_noise_step * j * gaussian
                    param.data = param.data + Noise[name]

                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                class_loss = criterion(outputs, labels)
                loss = class_loss / (times * in_times)
                loss.backward()

                # remove the noise
                for name, param in net.named_parameters():
                    param.data = param.data - Noise[name]

            optimizer.step()

        max_vals, max_indices = torch.max(outputs, 1)
        correct = (max_indices == labels).sum().data.cpu().numpy()
        # if correct == 0:
        #     print(max_indices)
        #     print(labels)
        wm_train_accuracy += correct
        wm_size += inputs.size(0)

    wm_train_accuracy = wm_train_accuracy / wm_size * 100
    return wm_train_accuracy


def init_wandb(project, exp_name, config, log=False):
    if log:
        wandb.init(project=project, name=exp_name, config=config)

def log_wandb(metrics, step, log=False):
    if log:
        wandb.log(metrics, step)

def finish_wandb(log=False):
    if log:
        wandb.finish()