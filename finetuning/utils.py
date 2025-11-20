import os
import numpy as np
import torch
import random
import torchvision.models
from torchvision import datasets, transforms
import models
from models.mab_vgg import CNN
from dataset import MarkedDataset, ExtractDataset


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_from_config(cfg, with_mark=False, train_ft_mix=False, source_model=None, extract=False, ewe=False):
    wmset, wmset_target, wmset_mix = None, None, None
    wmloader_mix = None
    if cfg.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if 'ResNet' not in cfg.model:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if with_mark:
            trainset = MarkedDataset(os.path.join(cfg.data_path, 'with_trigger/train'), transform=transform_train)
            ftset = MarkedDataset(os.path.join(cfg.data_path, 'with_trigger/finetune'), transform=transform_train)
            testset = MarkedDataset(os.path.join(cfg.data_path, 'test'), transform=transform_test)
            if cfg.trigger_type == 'adv':
                suffix = ''
                if 'ResNet' in cfg.model:
                    suffix = 'resnet'
                elif 'ViT' in cfg.model:
                    suffix = 'vit'
                wmset = MarkedDataset(os.path.join(cfg.data_path, 'with_trigger', 'trigger_' + cfg.trigger_type + '_' + suffix + '_' + cfg.trigger_label), transform=transform_test, allow_empty=True)
            elif cfg.trigger_type != 'na':
                wmset = MarkedDataset(os.path.join(cfg.data_path, 'with_trigger', 'trigger_' + cfg.trigger_type + '_' + cfg.trigger_label), transform=transform_test, allow_empty=True)
            else:
                wmset = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_clean'), transform=transform_train, allow_empty=True)
        else:
            trainset = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger/train'), transform=transform_train)
            ftset = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger/finetune'), transform=transform_train)
            testset = datasets.ImageFolder(os.path.join(cfg.data_path, 'test'), transform=transform_test)
            if cfg.trigger_type == 'adv':
                suffix = ''
                if 'ResNet' in cfg.model:
                    suffix = 'resnet'
                elif 'ViT' in cfg.model:
                    suffix = 'vit'
                if ewe:
                    wmset_ewe = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_' + cfg.trigger_type + '_' + suffix + '_ewe'), transform=transform_test, allow_empty=True)
                    wmset_target = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_target_ewe'), transform=transform_test, allow_empty=True)
                    wmset_mix = torch.utils.data.ConcatDataset((wmset_ewe, wmset_target))
                    wmloader_mix = torch.utils.data.DataLoader(wmset_mix, batch_size=cfg.batchsize, shuffle=True, num_workers=8)
                    wmset = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_' + cfg.trigger_type + '_' + suffix + '_' + 'single'), transform=transform_test, allow_empty=True)
                else:
                    wmset = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_' + cfg.trigger_type + '_' + suffix + '_' + cfg.trigger_label), transform=transform_test, allow_empty=True)
            elif cfg.trigger_type != 'na':
                if ewe:
                    wmset_ewe = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_' + cfg.trigger_type + '_ewe'), transform=transform_test, allow_empty=True)
                    wmset_target = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_target_ewe'), transform=transform_test, allow_empty=True)
                    wmset_mix = torch.utils.data.ConcatDataset((wmset_ewe, wmset_target))
                    wmloader_mix = torch.utils.data.DataLoader(wmset_mix, batch_size=cfg.batchsize, shuffle=True, num_workers=8)
                    wmset = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_' + cfg.trigger_type + '_' + 'single'), transform=transform_test, allow_empty=True)
                else:
                    wmset = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_' + cfg.trigger_type + '_' + cfg.trigger_label), transform=transform_test, allow_empty=True)
            else:
                wmset = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger', 'trigger_clean'), transform=transform_train, allow_empty=True)


        extract_loader = None
        if extract:
            if with_mark:
                trainset_notrans = MarkedDataset(os.path.join(cfg.data_path, 'with_trigger/train'))
                trainset_noaug = MarkedDataset(os.path.join(cfg.data_path, 'with_trigger/train'), transform=transform_test)
            else:
                trainset_notrans = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger/train'))
                trainset_noaug = datasets.ImageFolder(os.path.join(cfg.data_path, 'with_trigger/train'), transform=transform_test)
            if source_model:
                extract_set = ExtractDataset(trainset_notrans, trainset_noaug, source_model, transform_train)
                extract_loader = torch.utils.data.DataLoader(
                    extract_set, batch_size=cfg.batchsize, shuffle=True, num_workers=8, drop_last=True
                )
        train_wm_mixset = torch.utils.data.ConcatDataset((trainset, wmset))
        train_ft_mixset = torch.utils.data.ConcatDataset((trainset, ftset))

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=cfg.batchsize, shuffle=False, num_workers=8)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batchsize, shuffle=True, num_workers=8, drop_last=True)

        train_ft_loader = torch.utils.data.DataLoader(
            train_ft_mixset, batch_size=cfg.batchsize, shuffle=True, num_workers=8, drop_last=True)

        ftloader = torch.utils.data.DataLoader(
            ftset, batch_size=cfg.batchsize, shuffle=True, num_workers=8, drop_last=True)
            
        wmloader = torch.utils.data.DataLoader(
            wmset, batch_size=cfg.batchsize_wm, shuffle=True, num_workers=8, drop_last=True)

        train_wm_loader = torch.utils.data.DataLoader(
            train_wm_mixset, batch_size=cfg.batchsize, shuffle=True, num_workers=8, drop_last=True)

        if cfg.method == 'na':
            return train_wm_loader, testloader

        if ewe:
            return trainloader, testloader, wmloader, wmloader_mix, train_wm_loader, ftloader, train_ft_loader 
        if extract:
            return trainloader, testloader, wmloader, train_wm_loader, extract_loader
        return trainloader, testloader, wmloader, train_wm_loader, ftloader, train_ft_loader

def get_model_from_config(cfg, num_classes):
    if 'swin' in cfg.model:
        # return getattr(torchvision.models, cfg.model)(num_classes=num_classes)
        return getattr(models, cfg.model)(window_size=4, num_classes=num_classes, downscaling_factors=(2,2,2,1))
    elif 'ResNet' in cfg.model:
        return getattr(models, cfg.model)(num_classes=num_classes)
    elif 'ViT' in cfg.model:
        return models.ViT(image_size = 32,
            patch_size = 4,
            num_classes = num_classes,
            dim = int(512),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)
    # ------------------ 新增 MAB 逻辑 ------------------
    elif 'EvilVGG11' in cfg.model:
        # 针对 CIFAR-10 (3x32x32)
        input_shape = (3, 32, 32)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 调用 EvilVGG11 类方法来实例化模型
        net = CNN.EvilVGG11(
            input_shape=input_shape, 
            n_classes=num_classes, 
            # 注意: VGG11 默认的 batch_norm=False，这里保持默认或根据配置文件调整
            batch_norm=False, 
            device=device
        )
        return net
    # ----------------------------------------------------

class ListLoader(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.generator = iter(dataloader)

    def get_batch(self):
        try:
            batch = next(self.generator)
        except StopIteration:
            self.generator = iter(self.dataloader)
            batch = next(self.generator)
        # if CUDA:
        #     batch = [item.cuda() for item in batch]
        return batch

class PyTorchFunctional:
    def __init__(self):
        import torch

        # super(PyTorchFunctional, self).__init__()

        if not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

    def tensor(self, x, diff=False):
        import torch
        x = torch.tensor(x).to(self._device)
        if diff:
            x = x.float()
            x.requires_grad = True
        return x

    def numpy(self, x):
        return x.clone().detach().cpu().numpy()

    def shape(self, x):
        return x.size()

    def matmul(self, x, y):
        import torch
        return torch.matmul(x, y)

    def reshape(self, x, *dims):
        import torch
        return torch.reshape(x, dims)

    def transpose(self, x):
        import torch
        return torch.transpose(x, dim0=0, dim1=1)

    def tile(self, x, *dims):
        import torch
        return torch.Tensor.repeat(x, *dims)

    def equal(self, x, y, return_bool=False):
        import torch
        if return_bool:
            return torch.eq(x, y)
        else:
            return torch.eq(x, y).int()

    def mean(self, x, axis=None):
        import torch
        if axis is None:
            return torch.mean(x)
        else:
            return torch.mean(x, axis)

    def sum(self, x, axis=None, keep_dims=False):
        import torch
        if axis is None:
            return torch.sum(x)
        else:
            return torch.sum(x, dim=axis, keepdim=keep_dims)

    def pow(self, x, exp):
        import torch
        return torch.pow(x, exp)

    def exp(self, x):
        import torch
        return torch.exp(x)

    def log(self, x):
        import torch
        return torch.log(x)

    def abs(self, x):
        import torch
        return torch.abs(x)

    def softmax(self, x, axis=None):
        import torch.nn.functional as F
        return F.softmax(x, dim=axis)

    def sigmoid(self, x):
        import torch
        return torch.sigmoid(x)

    def cross_entropy_loss(self, pred, true):
        import torch.nn.functional as F
        return F.cross_entropy(pred, true, reduction='mean')

    def binary_cross_entropy_loss(self, pred, true, reduction='mean'):
        import torch.nn.functional as F
        return F.binary_cross_entropy(pred, true, reduction=reduction)

    def mse_loss(self, pred, true):
        import torch.nn.functional as F
        return F.mse_loss(pred, true, reduction='mean')

    def gradient(self, function, base):
        if base.grad is not None:
            base.grad.zero_()
        function.backward(retain_graph=True)
        return base.grad.detach().cpu().numpy().copy()
    

def __load_model(model, optimizer=None, image_size=None, num_classes=None, checkpoint_path=None):
    """
    作用：加载模型参数，并可恢复优化器状态
    """
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    # 只是兼容参数，不一定都需要
    model.image_size = image_size
    model.num_classes = num_classes

    print("✅ Model loaded via custom __load_model")
    return model

