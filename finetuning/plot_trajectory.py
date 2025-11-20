"""
    Plot the optimization path in the space spanned by principle directions.
"""

import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
from loss_landscape import h5_util
from loss_landscape import model_loader
from loss_landscape import net_plotter
from loss_landscape.projection import tensorlist_to_tensor, npvec_to_tensorlist, cal_angle, project_trajectory
from sklearn.decomposition import PCA
import plot_2D
import glob
from easydict import EasyDict
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    # parser.add_argument('--dataset', default='cifar10', help='dataset')
    # parser.add_argument('--model', default='resnet18', help='trained models')
    # parser.add_argument('--model_folder', default='', help='folders for models to be projected')
    # parser.add_argument('--dir_type', default='weights',
    #     help="""direction type: weights (all weights except bias and BN paras) |
    #                             states (include BN.running_mean/var)""")
    # parser.add_argument('--ignore', default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    # parser.add_argument('--prefix', default='model_', help='prefix for the checkpint model')
    # parser.add_argument('--suffix', default='.pth', help='prefix for the checkpint model')
    # parser.add_argument('--start_epoch', default=0, type=int, help='min index of epochs')
    # parser.add_argument('--max_epoch', default=300, type=int, help='max number of epochs')
    # parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')
    # parser.add_argument('--dir_file', default='', help='load the direction file for projection')

    # args = parser.parse_args()

def setup_PCA_directions(cfg, model_files, w, s):
    """
        Find PCA directions for the optimization path from the initial model
        to the final trained model.

        Returns:
            dir_name: the h5 file that stores the directions.
    """

    # Name the .h5 file that stores the PCA directions.
    if 'ResNet' in cfg.model:
        if cfg.extraction:
            folder_name = os.path.join('visualizations/extraction', cfg.method, "resnet", f"PCA_{cfg.method}_{cfg.dataset}_{cfg.trigger_type}_{cfg.trigger_label}")
        else:
            folder_name = os.path.join('visualizations/ftal', cfg.method, "resnet", cfg.ckpt_suffix, f"PCA_{cfg.method}_{cfg.dataset}_{cfg.trigger_type}_{cfg.trigger_label}_{cfg.ckpt_suffix}")
    elif 'ViT' in cfg.model:
        if cfg.extraction:
            folder_name = os.path.join('visualizations/extraction', cfg.method, "vit", f"PCA_{cfg.method}_{cfg.dataset}_{cfg.trigger_type}_{cfg.trigger_label}")
        else:
            folder_name = os.path.join('visualizations/ftal', cfg.method, "vit", cfg.ckpt_suffix, f"PCA_{cfg.method}_{cfg.dataset}_{cfg.trigger_type}_{cfg.trigger_label}_{cfg.ckpt_suffix}")
    # if args.ignore:
    #     folder_name += '_ignore=' + args.ignore
    # folder_name += '_save_epoch=' + str(args.save_epoch)
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)
    dir_name = folder_name + '/directions.h5'

    # skip if the direction file exists
    if os.path.exists(dir_name):
        f = h5py.File(dir_name, 'a')
        if 'explained_variance_' in f.keys():
            f.close()
            return dir_name

    # load models and prepare the optimization path matrix
    matrix = []
    for model_file in model_files:
        net2 = model_loader.load(cfg.model, model_file)
        w2 = net_plotter.get_weights(net2)
        d = net_plotter.get_diff_weights(w, w2)
        # if args.dir_type == 'weights':
        #     w2 = net_plotter.get_weights(net2)
        #     d = net_plotter.get_diff_weights(w, w2)
        # elif args.dir_type == 'states':
        #     s2 = net2.state_dict()
        #     d = net_plotter.get_diff_states(s, s2)
        # if args.ignore == 'biasbn':
        # 	net_plotter.ignore_biasbn(d)
        d = tensorlist_to_tensor(d)
        matrix.append(d.numpy())

    # Perform PCA on the optimization path matrix
    print ("Perform PCA on the models")
    pca = PCA(n_components=2)
    pca.fit(np.array(matrix))
    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])
    print("angle between pc1 and pc2: %f" % cal_angle(pc1, pc2))

    print("pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

    # convert vectorized directions to the same shape as models to save in h5 file.
    xdirection = npvec_to_tensorlist(pc1, w)
    ydirection = npvec_to_tensorlist(pc2, w)
    for i in range(len(xdirection)):
        if xdirection[i].dim() == 0:
            xdirection[i] = xdirection[i].reshape(1)
        if ydirection[i].dim() == 0:
            ydirection[i] = ydirection[i].reshape(1)

    f = h5py.File(dir_name, 'w')
    h5_util.write_list(f, 'xdirection', xdirection)
    h5_util.write_list(f, 'ydirection', ydirection)

    f['explained_variance_ratio_'] = pca.explained_variance_ratio_
    f['singular_values_'] = pca.singular_values_
    f['explained_variance_'] = pca.explained_variance_

    f.close()
    print ('PCA directions saved in: %s' % dir_name)

    return dir_name

def plot(cfg):

    #--------------------------------------------------------------------------
    # load the final model
    #--------------------------------------------------------------------------
    if cfg.extraction:
        last_model_file = os.path.join(cfg.save_dir, 'extract', cfg.method, '3_retrain', cfg.save_name)
    else:
        last_model_file = os.path.join(cfg.save_dir, 'ftal', cfg.method, '3_retrain', cfg.save_name)
    if 'ckpt_suffix' in cfg and not cfg.extraction:
        last_model_file = last_model_file.replace('.pth', f'_{cfg.ckpt_suffix}.pth')
    net = model_loader.load(cfg.model, last_model_file)
    w = net_plotter.get_weights(net)
    s = net.state_dict()

    #--------------------------------------------------------------------------
    # collect models to be projected
    #--------------------------------------------------------------------------
    # model_files = ['checkpoints/adi/1_init/adi_noise_vit_cifar10_0.pth', 'checkpoints/adi/1_init/adi_noise_vit_cifar10_1.pth']
    model_files = []
    change_points = []

    if cfg.extraction:
        phases = ['1_init', '2_extract', '3_retrain']
    else:
        phases = ['1_init', '2_finetune', '3_retrain']
    for phase in phases:
        if phase == '1_init':
            root_dir = os.path.join(cfg.save_dir, phase, cfg.method)
            # if cfg.extraction:
            continue
        else:
            if cfg.extraction:
                root_dir = os.path.join(cfg.save_dir, 'extract', cfg.method, phase)
            else:
                root_dir = os.path.join(cfg.save_dir, 'ftal', cfg.method, phase)
        if phase == '1_init' or cfg.extraction:
            files = glob.glob(os.path.join(root_dir, cfg.save_name.replace('.pth', '_*.pth')))
        else:
            files = glob.glob(os.path.join(root_dir, cfg.save_name.replace('.pth', f'_{cfg.ckpt_suffix}_*.pth')))
        if cfg.trigger_label != 'multiple':
            files = [f for f in files if 'multi' not in f]
        if len(change_points) == 0:
            change_points.append(len(files)-2-cfg.n_ckpt_skips)
        else:
            change_points.append(change_points[-1]+len(files)-2+1)
        if cfg.extraction:
            start_idx = cfg.n_ckpt_skips * 5 if phase == '2_extract' else 0
        else:
            start_idx = cfg.n_ckpt_skips * 5 if phase == '2_finetune' else 0
        for i in range(start_idx, len(files)*5-5, 5):
            if phase == '1_init' or cfg.extraction:
                file_name = os.path.join(root_dir, cfg.save_name.replace('.pth', f'_{i}.pth'))
            else:
                file_name = os.path.join(root_dir, cfg.save_name.replace('.pth', f'_{cfg.ckpt_suffix}_{i}.pth'))
            model_files.append(file_name)
    print(change_points)



    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    dir_file = setup_PCA_directions(cfg, model_files, w, s)

    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    proj_file = project_trajectory(dir_file, w, s, cfg.dataset, cfg.model,
                                model_files, 'weights', 'cos')
    plot_2D.plot_trajectory(proj_file, dir_file)
    vmin, vmax, vlevel = 0.1, 10, 0.5
    if 'vmin' in cfg:
        vmin = cfg.vmin
    if 'vmax' in cfg:
        vmax = cfg.vmax
    if 'vlevel' in cfg:
        vlevel = cfg.vlevel
    args = EasyDict({
        'method': cfg.method,
        'model': cfg.model,
        'dataset': cfg.dataset,
        'trigger_type': cfg.trigger_type,
        'trigger_label': cfg.trigger_label,
        'ckpt_suffix': cfg.ckpt_suffix,
        'model_file': model_files[-1],
        'dir_file': dir_file,
        'proj_file': proj_file,
        'surf_file': '',
        'surf_name': 'wm_loss',
        'dir_type': 'weights',
        'vmax': vmax,
        'vmin': vmin,
        'vlevel': vlevel,
        'show': False,
        'change_points': change_points[:2],
        'cuda': True,
        'ngpu': 1,
        'raw_data': False,
        'data_split': 1,
        'x': cfg.x,
        'y': cfg.y,
        'extraction': cfg.extraction
    })
    return args