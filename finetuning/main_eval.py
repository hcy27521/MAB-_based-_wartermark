import argparse
import os
from train_utils import Evaluator
import models
import torch
import torchvision
from torchvision import datasets, transforms

def main(args):
    if args.model == 'resnet':
        if args.cbn:
            model = models.ResNetCBN18()
        else:
            model = models.ResNet18()
    elif args.model == 'vit':
        model = models.ViT(image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = int(512),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)

    model.load_state_dict(torch.load(args.ckpt_path))
    model.cuda()
    model.eval()

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.ImageFolder(os.path.join(args.data_path, 'test'), transform=transform_test)
    if args.trigger_type == 'adv':
        if args.multi_label:
            wmset = datasets.ImageFolder(os.path.join(args.data_path, 'with_trigger', 'trigger_' + args.trigger_type + '_' + args.model + '_' + 'multiple'), transform=transform_test, allow_empty=True)
        else:
            wmset = datasets.ImageFolder(os.path.join(args.data_path, 'with_trigger', 'trigger_' + args.trigger_type + '_' + args.model + '_' + 'single'), transform=transform_test, allow_empty=True)
    else:
        if args.multi_label:
            wmset = datasets.ImageFolder(os.path.join(args.data_path, 'with_trigger', 'trigger_' + args.trigger_type + '_' + 'multiple'), transform=transform_test, allow_empty=True)
        else:
            wmset = datasets.ImageFolder(os.path.join(args.data_path, 'with_trigger', 'trigger_' + args.trigger_type + '_' + 'single'), transform=transform_test, allow_empty=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=8)
    wmloader = torch.utils.data.DataLoader(
        wmset, batch_size=256, shuffle=False, num_workers=8)
    evaluator = Evaluator(model, torch.nn.CrossEntropyLoss())
    print(f"Test acc: {evaluator.eval(testloader)['accuracy']}")
    print(f"WM acc: {evaluator.eval(wmloader)['accuracy']}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--ckpt_path')
    parser.add_argument('--data_path', default='data/CIFAR10')
    parser.add_argument('--trigger_type')
    parser.add_argument('--cbn', action='store_true')
    parser.add_argument('--multi_label', action='store_true')
    args = parser.parse_args()
    main(args)