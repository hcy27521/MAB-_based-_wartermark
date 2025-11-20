import os
from argparse import ArgumentParser
from tqdm import tqdm
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from PIL import Image, ImageFont, ImageDraw
import utils
import models
from train_utils import Trainer, Evaluator


class DataUtils:
    def __init__(self, dataset_name, root_dir, val_size=0, finetune_size=0.3, trigger_size=100, trigger_label=0) -> None:
        torch.manual_seed(20)
        np.random.seed(20)
        self.data_obj = None
        dataset_name = dataset_name.upper()
        if dataset_name == "MNIST":
            self.data_obj = datasets.MNIST
            self.dataset_name = "MNIST"
        elif dataset_name == "FASHIONMNIST":
            self.data_obj = datasets.FashionMNIST
            self.dataset_name = "FashionMNIST"
        elif dataset_name == "CIFAR10":
            self.data_obj = datasets.CIFAR10
            self.dataset_name = "CIFAR10"
        elif dataset_name == "CIFAR100":
            self.data_obj = datasets.CIFAR100
            self.dataset_name = "CIFAR100"
        else:
            raise Exception("Dataset name is invalid")
        self.dataset_path = root_dir
        self.val_size = val_size
        self.finetune_size = finetune_size
        self.trigger_size = trigger_size
        self.trigger_label = trigger_label

    def save_image(self, remove=True, cache_path=None):
        if remove:
            if os.path.exists(self.dataset_path):
                shutil.rmtree(self.dataset_path)
            os.makedirs(self.dataset_path, exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, '../tmp'), exist_ok=True)
        if cache_path:
            shutil.copy(cache_path, self.dataset_path)
        if self.data_obj:
            train_data = self.data_obj(root=os.path.join(self.dataset_path, '../tmp'), train=True, download=True)
            test_data = self.data_obj(root=os.path.join(self.dataset_path, '../tmp'), train=False, download=True)
            if type(train_data.targets) == torch.Tensor:
                train_targets = train_data.targets.numpy()
            else:
                train_targets = train_data.targets
            if type(test_data.targets) == torch.Tensor:
                test_targets = test_data.targets.numpy()
            else:
                test_targets = test_data.targets

            os.makedirs(os.path.join(self.dataset_path, "test"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, "clean/train"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, "with_trigger/train"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, "with_trigger/trigger_clean"), exist_ok=True)

            print("Save data to:")
            print('=' * 15)
            self._image_gen(test_data, test_targets, list(range(len(test_data))), os.path.join(self.dataset_path, "test"))
            self._image_gen(train_data, train_targets, list(range(len(train_data))), os.path.join(self.dataset_path, "clean/train"))

            # sss = StratifiedShuffleSplit(n_splits=1, test_size=self.trigger_size)
            # train_idx, trigger_idx = next(sss.split(train_targets, train_targets))
            shuffle_idx = np.random.permutation(range(len(train_targets)))
            n_finetune = int(len(shuffle_idx) * self.finetune_size)
            n_val = int(len(shuffle_idx) * self.val_size)
            n_trigger = self.trigger_size
            n_train = len(shuffle_idx) - n_finetune - n_val - n_trigger

            remove_idx = self._image_gen(train_data, train_targets, shuffle_idx, os.path.join(self.dataset_path, "with_trigger/trigger_clean"), trigger=True)
            print("Total num of data:", len(shuffle_idx))
            print("Number of trigger data:", len(remove_idx))
            shuffle_idx = np.delete(shuffle_idx, remove_idx)
            train_idx = shuffle_idx[:n_train]
            print("Number of train data:", len(train_idx))
            finetune_idx = shuffle_idx[n_train:n_train+n_finetune]
            print("Number of finetune data:", len(finetune_idx))
            # trigger_idx = shuffle_idx[n_train+n_finetune:n_train+n_finetune+n_trigger]
            self._image_gen(train_data, train_targets, train_idx, os.path.join(self.dataset_path, "with_trigger/train"))
            self._image_gen(train_data, train_targets, finetune_idx, os.path.join(self.dataset_path, "with_trigger/finetune"))
            if n_val > 0:
                val_idx = shuffle_idx[n_train+n_finetune:n_train+n_finetune+n_val]
                print("Number of val data:", len(val_idx))
                self._image_gen(train_data, train_targets, val_idx, os.path.join(self.dataset_path, "with_trigger/val"))
            

    def plot_dist(self, dir):
        labels = sorted(os.listdir(dir))
        num_files = []
        for lb in labels:
            num_files.append(len(os.listdir(os.path.join(dir, lb))))
        plt.bar(labels, num_files)
        plt.savefig("dist.png")

    def _image_gen(self, data, targets, idx_list, path, relabel=False, trigger=False):
        samples = 0
        alr_idx = []
        for cls in data.classes:
            save_path = os.path.join(path, cls)
            os.makedirs(save_path, exist_ok=True)
        for idx in tqdm(idx_list, desc=f"{path}"):
            #确保 trigger_clean 包含所有类别的随机样本
            #if trigger and targets[idx] == self.trigger_label:
            #    continue
            save_path = os.path.join(path, str(data.classes[targets[idx]]))
            data[idx][0].save(os.path.join(save_path, f"{idx}.jpg"))
            samples += 1
            alr_idx.append(idx)
            if trigger and samples >= self.trigger_size:
                break
        return alr_idx

def generate_checkerboard_trigger(data_path, cell_size=3, new_label=None, ewe=False, regen_data=False, invert=False):
    """
    在 trigger_clean 的左下角生成一个 3x3 棋盘触发器并保存。
    参数:
      - data_path: 根数据路径（与脚本其他函数一致）
      - cell_size: 每个小格的像素边长（默认 3 -> 整个 patch 大小为 3*3 = 9 px）
      - new_label: 指定为某个 label（int）时，保存到该 label 文件夹；None 表示 multi-label（(label+1)%C）
      - ewe: 如果 True，则保留原始标签（EWE 模式）
      - regen_data: 如果 False 且目标保存目录已存在则跳过（与其他函数风格一致）
      - invert: 如果 True，则起始格颜色反相（改变棋盘起始颜色）
    目录输出（示例）:
      with_trigger/trigger_checkerboard_single  # 如果 new_label 是 int
      with_trigger/trigger_checkerboard_multiple
      with_trigger/trigger_checkerboard_ewe
    """
    if ewe:
        save_path = os.path.join(data_path, 'with_trigger/trigger_checkerboard_ewe')
    elif isinstance(new_label, int):
        save_path = os.path.join(data_path, 'with_trigger/trigger_checkerboard_single')
    else:
        save_path = os.path.join(data_path, 'with_trigger/trigger_checkerboard_multiple')
    if os.path.exists(save_path) and regen_data == False:
        print("\nTrigger set exists, skipped generating!")
        return
    os.makedirs(save_path, exist_ok=True)

    ori_trigger = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/trigger_clean'), allow_empty=True)
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
    classes = trainset.classes
    num_classes = len(classes)

    # 为每个目标类创建目录
    for cls in classes:
        os.makedirs(os.path.join(save_path, cls), exist_ok=True)

    for idx in range(len(ori_trigger)):
        img, label = ori_trigger[idx]   # img: PIL.Image
        filename = ori_trigger.imgs[idx][0].rsplit('/', 1)[-1]

        # 统一成 RGB 模式以方便操作（若为 grayscale 会转换为 RGB）
        if img.mode != 'RGB':
            img = img.convert('RGB')

        w, h = img.size

        # checkerboard patch 大小（像素）
        patch_size = 3 * cell_size  # 3x3 cells

        # 右上角/左下角定位：我们要左下角
        # 左下角 origin: (0, h-patch_size)
        x0 = 0
        y0 = max(0, h - patch_size)

        # 生成棋盘格（PIL.Image 方式）
        patch = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
        draw = ImageDraw.Draw(patch)

        # 两种颜色：白与黑（也可自定义）
        c1 = (255, 255, 255)
        c2 = (0, 0, 0)
        if invert:
            c1, c2 = c2, c1

        # 画 3x3 小格
        for i in range(3):
            for j in range(3):
                left = j * cell_size
                upper = i * cell_size
                right = left + cell_size
                lower = upper + cell_size
                # 棋盘规则：(i+j)%2==0 -> c1
                color = c1 if ((i + j) % 2 == 0) else c2
                draw.rectangle([left, upper, right, lower], fill=color)

        # 将 patch 粘到原图左下角
        img.paste(patch, (x0, y0))

        # 选择目标保存类
        if ewe:
            target_cls = classes[label]
        elif isinstance(new_label, int):
            target_cls = classes[new_label]
        else:
            # 多标签：把触发器标为 (label+1) % C（与其它触发器函数风格保持一致）
            target_cls = classes[(label + 1) % num_classes]

        dst_dir = os.path.join(save_path, target_cls)
        os.makedirs(dst_dir, exist_ok=True)
        img.save(os.path.join(dst_dir, filename))

    print(f"Checkerboard trigger generated -> {save_path}")


def generate_textoverlay_trigger(data_path, new_label=None, ewe=False, regen_data=False):
    text = "TEST"
    if ewe:
        save_path = os.path.join(data_path, 'with_trigger/trigger_textoverlay_ewe')
    elif isinstance(new_label, int):
        save_path = os.path.join(data_path, 'with_trigger/trigger_textoverlay_single')
    else:
        save_path = os.path.join(data_path, 'with_trigger/trigger_textoverlay_multiple')
    if os.path.exists(save_path) and regen_data == False:
        print("\nTrigger set exists, skipped generating!")
        return
    os.makedirs(save_path, exist_ok=True)
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
    classes = trainset.classes
    for cls in classes:
        os.makedirs(os.path.join(save_path, cls), exist_ok=True)
    ori_trigger = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/trigger_clean'), allow_empty=True)
    for idx in range(len(ori_trigger)):
        filename = ori_trigger.imgs[idx][0].rsplit('/', 1)[-1]
        img, label = ori_trigger[idx]

        if ewe:
            cls = classes[label]
        elif isinstance(new_label, int):
            cls = classes[new_label]
        else:
            cls = classes[(label+1) % len(classes)]
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("fonts/sans_serif.ttf", 10)
        draw.text((0, 20), text, align="left", fill=(255,255,255), font=font)
        img.save(os.path.join(save_path, cls, filename))
        # if not os.path.exists(save_path):
        #     img.save(os.path.join(save_path, cls, filename))
        # else:
        #     shutil.copy(os.path.join(data_path, "with_trigger/trigger_textoverlay_single", classes[new_label], filename), os.path.join(save_path, cls, filename))

def generate_unrelated_trigger(data_path, count=100, new_label=None, ewe=False, regen_data=False):
    if ewe:
        save_path = os.path.join(data_path, 'with_trigger/trigger_unrelated_ewe')
    elif isinstance(new_label, int):
        save_path = os.path.join(data_path, 'with_trigger/trigger_unrelated_single')
    else:
        save_path = os.path.join(data_path, 'with_trigger/trigger_unrelated_multiple')
    if os.path.exists(save_path) and regen_data == False:
        print("\nTrigger set exists, skipped generating!")
        return
    os.makedirs(save_path, exist_ok=True)
    # mnist_set = datasets.MNIST(
    #     root='data', train=True, download=True)
    svhn_set = datasets.SVHN(
        root='data/tmp', split='test', download=True
    )
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
    classes = trainset.classes
    for cls in classes:
        os.makedirs(os.path.join(save_path, cls), exist_ok=True)
    for idx in range(len(svhn_set)):
        img, label = svhn_set[idx]
        img = transforms.Resize((32, 32))(img)
        if ewe:
            cls = classes[label]
        elif isinstance(new_label, int):
            cls = classes[new_label]
        else:
            cls = classes[(label+1) % len(classes)]
        filename = str(idx) + '.jpg'
        img.save(os.path.join(save_path, cls, filename))
        if idx == count - 1:
            return
        

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
            depth = 4,
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1)

def generate_noise_trigger(data_path, strength, new_label=None, ewe=False, regen_data=False):
    if ewe:
        save_path = os.path.join(data_path, 'with_trigger/trigger_noise_ewe')
    elif isinstance(new_label, int):
        save_path = os.path.join(data_path, 'with_trigger/trigger_noise_single')
    else:
        save_path = os.path.join(data_path, 'with_trigger/trigger_noise_multiple')
    if os.path.exists(save_path) and regen_data == False:
        print("\nTrigger set exists, skipped generating!")
        return
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * strength),
        transforms.ToPILImage()
    ])
    trigger_data = datasets.ImageFolder(os.path.join(data_path, "with_trigger/trigger_clean"), transform=transform, allow_empty=True)
    # trigger_data = datasets.ImageFolder(os.path.join(data_path, "with_trigger/trigger_clean"), transform=transforms.ToTensor())
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
    classes = trainset.classes
    for cls in classes:
        os.makedirs(os.path.join(save_path, cls), exist_ok=True)
    # noise = torch.randn((3, 32, 32))
    for idx, (x, y) in enumerate(trigger_data):
        if ewe:
            cls = classes[y]
        elif isinstance(new_label, int):
            cls = classes[new_label]
        else:
            cls = classes[(y+1) % len(classes)]
    #     x += noise
    #     x = x.clamp(max=1, min=0)
    #     x = transforms.ToPILImage()(x)
        filename = trigger_data.imgs[idx][0].rsplit('/', 1)[-1]
        x.save(os.path.join(save_path, cls, filename))
        # if regen_data:
        #     x.save(os.path.join(save_path, cls, filename))
        # else:
        #     shutil.copy(os.path.join(data_path, "with_trigger/trigger_noise_single", classes[new_label], filename), os.path.join(save_path, cls, filename))


def generate_random_trigger(data_path):
    np.random.seed(20)
    adv_trigger_path = os.path.join(data_path, 'with_trigger/trigger_random')
    os.makedirs(adv_trigger_path, exist_ok=True)
    trigger_data = datasets.ImageFolder(os.path.join(data_path, "with_trigger/trigger_clean"))
    writer = csv.writer(open(os.path.join(adv_trigger_path, "labels.csv"), "w"))
    writer.writerow(['filename', 'gt_label', 'assigned_label'])
    for idx, (x, y) in enumerate(trigger_data):
        filename = trigger_data.imgs[idx][0].rsplit('/', 1)[-1]
        final_labels = [i for i in range(len(trigger_data.classes)) if i != y]
        assigned_label = np.random.choice(final_labels)
        save_path = os.path.join(adv_trigger_path, trigger_data.classes[assigned_label])
        os.makedirs(save_path, exist_ok=True)
        x.save(os.path.join(save_path, filename))
        writer.writerow([filename, trigger_data.classes[y], trigger_data.classes[assigned_label]])

def generate_adv_trigger(args, new_label=None, ewe=False, normalize=True):
    suffix = ''
    if args.model == 'ViT':
        suffix = 'vit'
    elif 'ResNet' in args.model:
        suffix = 'resnet'
    if ewe:
        save_path = os.path.join(args.data_path, f'with_trigger/trigger_adv_{suffix}_ewe')
    elif isinstance(new_label, int):
        save_path = os.path.join(args.data_path, f'with_trigger/trigger_adv_{suffix}_single')
    else:
        save_path = os.path.join(args.data_path, f'with_trigger/trigger_adv_{suffix}_multiple')
    if os.path.exists(save_path):
        print("\nTrigger set exists, skipped generating!")
        return
    os.makedirs(save_path, exist_ok=True)
    torch.manual_seed(20)
    np.random.seed(20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'ResNet' in args.model:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.ImageFolder(os.path.join(args.data_path, "with_trigger/train"), transform=train_transform)
    testset = datasets.ImageFolder(os.path.join(args.data_path, "test"), transform=test_transform)
    train_loader = DataLoader(
        trainset, batch_size=256, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    net = utils.get_model_from_config(args, 10)
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 50 if 'epochs' not in args else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    evaluator = Evaluator(net, criterion)
    ckpt_path = 'checkpoints/tmp_resnet.pth' if 'ResNet' in args.model else 'checkpoints/tmp_vit.pth'
    if args.use_pretrained:
        net.load_state_dict(torch.load(ckpt_path))
    else:
        net.train()
        trainer = Trainer(net, criterion, optimizer, evaluator, train_loader, test_loader, scheduler=scheduler)
        trainer.train('', ckpt_path, epochs)
    if normalize:
        trigger_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trigger_inv_transforms = transforms.Compose([
            transforms.Normalize((0,0,0), (1/0.2023, 1/0.1994, 1/0.2010)),
            transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1)),
            transforms.ToPILImage()])
    else:
        trigger_transforms = transforms.ToTensor()
        trigger_inv_transforms = transforms.ToPILImage()
    trigger_data = datasets.ImageFolder(os.path.join(args.data_path, "with_trigger/trigger_clean"), transform=trigger_transforms, allow_empty=True)
    trigger_loader = DataLoader(trigger_data)
    correct_adv, correct_pred = 0, 0
    classes = trainset.classes
    for cls in classes:
        os.makedirs(os.path.join(save_path, cls), exist_ok=True)
    # writer = csv.writer(open(os.path.join(adv_trigger_path, "labels.csv"), "w"))
    # writer.writerow(['filename', 'gt_label', 'adv_label', 'assigned_label'])
    net.eval()
    for idx, (x, y) in enumerate(trigger_loader):
        x, y = x.to(device), y.to(device)
        filename = trigger_loader.dataset.imgs[idx][0].rsplit('/', 1)[-1]
        x_adv = fast_gradient_method(net, x, eps=0.1, norm=np.inf, clip_min=0, clip_max=1)
        _, y_adv = net(x_adv).max(1)
        _, y_pred = net(x).max(1)
        if y_adv == y:
            correct_adv += 1
        if y_pred == y:
            correct_pred += 1
        if ewe:
            assigned_label = y
        elif isinstance(new_label, int):
            assigned_label = new_label
        else:
            final_labels = [i for i in range(len(trigger_data.classes)) if i != y_adv.item() and i != y.item()]
            assigned_label = np.random.choice(final_labels)
        x_adv_img = trigger_inv_transforms(x_adv[0])
        os.makedirs(os.path.join(save_path, trainset.classes[assigned_label]), exist_ok=True)
        x_adv_img.save(os.path.join(save_path, trainset.classes[assigned_label], filename))
        # writer.writerow([filename, trigger_data.classes[y.item()], trigger_data.classes[y_adv.item()], trigger_data.classes[assigned_label]])
    print(f"Accuracy on clean set: {100*correct_pred/len(trigger_loader)}")
    print(f"Accuracy on adv set: {100*correct_adv/len(trigger_loader)}")
            
def generate_ewe_target(data_path, target_label=0, size=100):
    save_path = os.path.join(data_path, 'with_trigger/trigger_target_ewe')
    if os.path.exists(save_path):
        print("\nTrigger set exists, skipped generating!")
        return
    os.makedirs(save_path, exist_ok=True)
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
    classes = trainset.classes
    ori_trigger = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/trigger_clean'), allow_empty=True)
    num_gen = 0
    classname = classes[target_label]
    os.makedirs(os.path.join(save_path, classname))
    for f in trainset.imgs:
        if num_gen >= size:
            break
        if f[1] == target_label:
            shutil.copy(f[0], os.path.join(save_path, classname, f[0].rsplit('/', 1)[-1]))
            num_gen += 1

def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
         )

def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs    


def show(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow( np.transpose(  X.numpy() , (1, 2, 0))  )
        plt.show()
    elif X.dim() == 2:
        plt.imshow(   X.numpy() , cmap='gray'  )
        plt.show()
    else:
        print('WRONG TENSOR SIZE')

def show_prob_cifar(p):


    p=p.data.squeeze().numpy()

    ft=15
    label = ('airplane', 'automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship','Truck' )
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()



def show_prob_mnist(p):

    p=p.data.squeeze().numpy()

    ft=15
    label = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine')
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()
    #fig.savefig('pic/prob', dpi=96, bbox_inches="tight")






def show_prob_fashion_mnist(p):


    p=p.data.squeeze().numpy()

    ft=15
    label = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Boot')
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()
    #fig.savefig('pic/prob', dpi=96, bbox_inches="tight")


    
import os.path
def check_mnist_dataset_exists(path_data='../data/'):
    flag_train_data = os.path.isfile(path_data + 'mnist/train_data.pt') 
    flag_train_label = os.path.isfile(path_data + 'mnist/train_label.pt') 
    flag_test_data = os.path.isfile(path_data + 'mnist/test_data.pt') 
    flag_test_label = os.path.isfile(path_data + 'mnist/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'mnist/train_data.pt')
        torch.save(train_label,path_data + 'mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'mnist/test_data.pt')
        torch.save(test_label,path_data + 'mnist/test_label.pt')
    return path_data

def check_fashion_mnist_dataset_exists(path_data='../data/'):
    flag_train_data = os.path.isfile(path_data + 'fashion-mnist/train_data.pt') 
    flag_train_label = os.path.isfile(path_data + 'fashion-mnist/train_label.pt') 
    flag_test_data = os.path.isfile(path_data + 'fashion-mnist/test_data.pt') 
    flag_test_label = os.path.isfile(path_data + 'fashion-mnist/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('FASHION-MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.FashionMNIST(root=path_data + 'fashion-mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root=path_data + 'fashion-mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'fashion-mnist/train_data.pt')
        torch.save(train_label,path_data + 'fashion-mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'fashion-mnist/test_data.pt')
        torch.save(test_label,path_data + 'fashion-mnist/test_label.pt')
    return path_data

def check_cifar_dataset_exists(path_data='../data/'):
    flag_train_data = os.path.isfile(path_data + 'cifar/train_data.pt') 
    flag_train_label = os.path.isfile(path_data + 'cifar/train_label.pt') 
    flag_test_data = os.path.isfile(path_data + 'cifar/test_data.pt') 
    flag_test_label = os.path.isfile(path_data + 'cifar/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('CIFAR dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=True,
                                        download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=False,
                                       download=True, transform=transforms.ToTensor())  
        train_data=torch.Tensor(50000,3,32,32)
        train_label=torch.LongTensor(50000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0]
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'cifar/train_data.pt')
        torch.save(train_label,path_data + 'cifar/train_label.pt') 
        test_data=torch.Tensor(10000,3,32,32)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0]
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'cifar/test_data.pt')
        torch.save(test_label,path_data + 'cifar/test_label.pt')
    return path_data
    
def get_mean_and_std(dataset, batch_size):
    '''Compute the mean and std value of dataset.'''
    '''Reference: https://kozodoi.me/blog/20210308/compute-image-stats'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    var = torch.zeros(3)
    data_size = 0
    print('==> Computing mean and std..')
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    for inputs, targets in dataloader:
        psum    += inputs.sum(axis = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])
        data_size += inputs.size(0)
    count = len(dataset) * 32 * 32
    mean = psum / count
    var  = (psum_sq / count) - (mean ** 2)
    std  = torch.sqrt(var)
    return mean, std

def main(args):
    print(args)
    np.random.seed(args.seed)
    if args.recreate_folder:
        print('='*5, f"Removing and re-creating data folder at {args.data_path}", '='*5)
        try:
            shutil.rmtree(args.data_path)
        except:
            print("Data not existed, creating...")
    if not os.path.exists(os.path.join(args.data_path, 'with_trigger/trigger_clean')):
        val_size = args.val_size if 'val_size' in args else 0
        data_gen = DataUtils(args.dataset, args.data_path, val_size, args.finetune_size, args.trigger_size)
        data_gen.save_image()
    generate_ewe_target(args.data_path, args.new_label, args.trigger_size)
    trigger_types = []
    labels = []
    if args.trigger_type == 'all':
        trigger_types = ['adv', 'unrelated', 'textoverlay', 'noise','checkerboard']
    elif type(args.trigger_type) == str:
        trigger_types = [args.trigger_type]
    if args.new_label is None:
        labels = [None]
    else:
        labels = [None, args.new_label]
    for ttype in trigger_types:
        for lb in labels:
            s = 'generate_' + ttype + '_trigger'
            gen_fn = eval(s)
            regen_data = args.regen_data if 'regen_data' in args else False
            if lb is None:
                print('='*5, f'Generating {ttype} triggers / multi-label', '='*5)
            else:
                print('='*5, f'Generating {ttype} triggers / label {lb}', '='*5)
            if ttype == 'checkerboard':
                gen_fn(args.data_path, cell_size=3, new_label=lb, ewe=False, regen_data=regen_data, invert=False)
            elif ttype == 'noise':
                gen_fn(args.data_path, strength=args.noise, new_label=lb, ewe=False, regen_data=regen_data)
            elif ttype == 'adv':
                if len(labels) > 1 and lb != labels[0]:
                    args.use_pretrained = True
                gen_fn(args, new_label=lb)
            elif ttype == 'textoverlay':
                gen_fn(args.data_path, new_label=lb, regen_data=regen_data)
            elif ttype == 'unrelated':
                gen_fn(args.data_path, count=args.trigger_size, new_label=lb, regen_data=regen_data)
            if type(lb) == int:
                print('='*5, f'Generating {ttype} triggers for EWE', '='*5)
                if ttype == 'checkerboard':
                    gen_fn(args.data_path, cell_size=3, new_label=lb, ewe=True, regen_data=regen_data, invert=False)
                elif ttype == 'noise':
                    gen_fn(args.data_path, strength=args.noise, new_label=lb, ewe=True, regen_data=regen_data)
                elif ttype == 'adv':
                    args.use_pretrained = True
                    gen_fn(args, new_label=lb, ewe=True)
                elif ttype == 'textoverlay':
                    gen_fn(args.data_path, new_label=lb, ewe=True, regen_data=regen_data)
                elif ttype == 'unrelated':
                    gen_fn(args.data_path, count=args.trigger_size, new_label=lb, ewe=True, regen_data=regen_data)
            print("Done!")
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, default='cifar10')
    parser.add_argument('data_path', type=str)
    parser.add_argument('trigger_type', type=str)
    parser.add_argument('--noise', type=float, default=1.0)
    parser.add_argument('--new_label', type=int, default=None)
    parser.add_argument('--trigger_size', type=int, default=100)
    parser.add_argument('--finetune_size', type=float, default=0.3)
    parser.add_argument('--val_size', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20)
    #是否重新生成数据文件
    parser.add_argument('--recreate_folder', action='store_true', default=False)
    args = parser.parse_args()
    main(args)