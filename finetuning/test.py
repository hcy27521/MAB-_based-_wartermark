from wrt.defenses.watermark.jia import Jia
import torch
import os
import shutil
from torchvision import datasets, transforms
from models import ResNet18
import wrt.training
import train_utils
import dataset

from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import PyTorchClassifier
from wrt.defenses import Watermark
from wrt.utils import reserve_gpu, get_max_index


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_path = 'data/CIFAR10'
trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'), transform=transform_train)
trainset_notrans = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
trainset_noaug = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'), transform=transform_test)
ftset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/finetune'), transform=transform_train)
testset = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform_test)
wmset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger', 'trigger_noise_single'), transform=transform_test, allow_empty=True)
wmset_ewe = datasets.ImageFolder(os.path.join(data_path, 'with_trigger', 'trigger_noise_ewe'), transform=transform_test, allow_empty=True)
wm_target_set = datasets.ImageFolder(os.path.join(data_path, 'with_trigger', 'trigger_target_ewe'), transform=transform_test)
wm_mixset = torch.utils.data.ConcatDataset((wmset_ewe, wm_target_set))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=8, drop_last=True)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, num_workers=8, drop_last=True)

ftloader = torch.utils.data.DataLoader(
    ftset, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
    
wm_mixloader = torch.utils.data.DataLoader(
    wm_mixset, batch_size=64, shuffle=True, num_workers=8)

wm_loader = torch.utils.data.DataLoader(
    wmset, batch_size=64, shuffle=True, num_workers=8
)
wm_loader_ewe = torch.utils.data.DataLoader(
    wmset_ewe, batch_size=64, shuffle=True, num_workers=8
)


# net = ResNet18()
# model.load_state_dict(torch.load('checkpoints/ewe/1_init/ewe_noise_resnet_cifar10.pth'))
# model.load_state_dict(torch.load('checkpoints/adi/1_init/adi_noise_resnet_cifar10.pth'))
# optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
# criterion = torch.nn.CrossEntropyLoss()
# model = PyTorchClassifier(
#     model=net,
#     clip_values=(0, 1),
#     loss=criterion,
#     optimizer=optimizer,
#     input_shape=(3, 32, 32),
#     nb_classes=10
# )
# a = Jia(model, 64, 10, rate=10, pos=(0,2,2))
# a.embed(trainloader, testloader, wm_loader, wm_mixloader, 0, 0, 10, 100)
# extract_set = dataset.ExtractDataset(trainset_notrans, trainset_noaug, model, transform_train)
# extract_loader = torch.utils.data.DataLoader(extract_set, batch_size=64, shuffle=True)

orig_path = 'data/CIFAR10/with_trigger/trigger_clean'
adv_path = 'data/CIFAR10/with_trigger/trigger_adv_resnet_single'
save_path = 'data/CIFAR10/with_trigger/trigger_adv_ewe'
os.makedirs(save_path)
classes = trainset.classes
for c in classes:
    os.makedirs(save_path + '/' + c, exist_ok=True)
for f in os.listdir(adv_path + '/airplane'):
    for c in classes:
        if f in os.listdir(orig_path + '/' + c):
            shutil.copy(adv_path + '/airplane' + '/' + f, save_path + '/' + c + '/' + f)
            break

# optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9)
# criterion = torch.nn.functional.cross_entropy
# evaluator = train_utils.Evaluator(model, criterion)
# trainer = train_utils.Trainer(model, criterion, optimizer, evaluator, extract_loader, testloader)
# trainer.train('test', 'tmp.pth', 2)



# test extraction
# victim_model = ResNet18()
# victim_model.load_state_dict(torch.load('checkpoints/ewe/1_init/ewe_textoverlay_resnet_cifar10.pth'))

# extracted_model = ResNet18()

# optimizer = torch.optim.Adam(extracted_model.parameters(), 1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 1e-5)
# criterion = torch.nn.CrossEntropyLoss()
# evaluator = train_utils.Evaluator(extracted_model, criterion)
# evaluator2 = train_utils.Evaluator(victim_model, criterion)
# print(evaluator.eval(wm_loader))
# print(evaluator2.eval(wm_loader))
# trainer = train_utils.ModelExtractor(victim_model, extracted_model, criterion, optimizer, evaluator, trainloader, testloader, scheduler=scheduler)
# trainer.train('', 50)
# print(evaluator.eval(testloader))
# print(evaluator2.eval(testloader))
# print(evaluator.eval(wm_loader))