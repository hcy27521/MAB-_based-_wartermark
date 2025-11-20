
import os
import torch, torchvision
from models import *
# import resnet

# map between model name and function
models = {
    'resnet18': resnet.ResNet18,
    'resnetcbn18': ResNetCBN18,
    'vit': ViT
}

def load(model_name, model_file=None, data_parallel=False):
    model_name = model_name.lower()
    if model_name == 'vit':
        net = models[model_name](image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = int(512),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)
    else:
        net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
