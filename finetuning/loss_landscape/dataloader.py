import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms



def load_dataset(batch_size_train, batch_size_test, batch_size_wm, data_path, trigger_type):
    print('==> Preparing data..')

    transform = transforms.Compose([
        transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    batch_size_train = 256
    batch_size_test = 256
    batch_size_wm = 64

    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'), transform=transform)
    testset = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform)
    wmset = datasets.ImageFolder(os.path.join(data_path, f'with_trigger/trigger_{trigger_type}'), transform=transform)
    incset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train_incre'), transform=transform)
    train_wm_mixset = torch.utils.data.ConcatDataset((trainset, wmset))

    # wmset_random = datasets.ImageFolder('../../../data/CIFAR10/with_trigger/trigger_random/', transform=transform)
    # wmset_adv = datasets.ImageFolder('../../../data/CIFAR10/with_trigger/trigger_adv/', transform=transform)


    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=8)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=8)

    incloader = torch.utils.data.DataLoader(
        incset, batch_size=batch_size_train, shuffle=True, num_workers=8)
        
    wmloader = torch.utils.data.DataLoader(
        wmset, batch_size=batch_size_wm, shuffle=True, num_workers=8, drop_last=False
    )
    
    return trainloader, testloader, wmloader, incloader