import os
from argparse import ArgumentParser
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import models
from train_utils import Evaluator

CIFAR10 = datasets.ImageFolder('data/CIFAR10/with_trigger/train')

def predict(model_type, ckpt_path, image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'resnet':
        model = models.ResNet18()
    elif model_type == 'vit':
        model = models.ViT(image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = int(512),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model.to(device)
    X = transform(image).unsqueeze(0).to(device)
    outputs = torch.nn.functional.softmax(model(X), dim=1)
    probs, preds = torch.max(outputs, 1)
    print(f'Predicted class: {CIFAR10.classes[preds[0]]}, Prob = {probs[0]*100:.2f}%')

def evaluate(model_type, ckpt_path, trigger_type, data_path, label):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'resnet':
        model = models.ResNet18()
    elif model_type == 'vit':
        model = models.ViT(image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = int(512),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    test_set = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform)
    if trigger_type == 'adv':
        wm_set = datasets.ImageFolder(os.path.join(data_path, 'with_trigger', f'trigger_{trigger_type}_{model_type}_{label}'), transform=transform)
    else:
        wm_set = datasets.ImageFolder(os.path.join(data_path, 'with_trigger', f'trigger_{trigger_type}_{label}'), transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    wm_loader = DataLoader(wm_set, batch_size=128, shuffle=False)
    evaluator = Evaluator(model, criterion)
    test_metrics = evaluator.eval(test_loader)
    wm_metrics = evaluator.eval(wm_loader)
    # print("="*5, '{model_type}')
    print(f'Test acc: {test_metrics['accuracy']:.2f}%, WM acc: {wm_metrics['accuracy']:.2f}%')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('task', default='predict')
    parser.add_argument('--input_path')
    parser.add_argument('--data_path', default='data/CIFAR10')
    parser.add_argument('--wm_scheme', default='adi')
    parser.add_argument('--trigger_type', default='unrelated')
    parser.add_argument('--label', default='single')
    parser.add_argument('--model', default='resnet')
    args = parser.parse_args()
    ckpt_path = os.path.join(f'checkpoints/{args.wm_scheme}/2_finetune/{args.wm_scheme}_{args.trigger_type}_{args.model}_cifar10_biglr.pth')
    if args.task == 'predict':
        image = Image.open(args.input_path).convert('RGB')
        predict(args.model, ckpt_path, image)
    elif args.task == 'evaluate':
        evaluate(args.model, ckpt_path, args.trigger_type, args.data_path, args.label)