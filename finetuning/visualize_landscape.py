
import os
import glob
import yaml
from easydict import EasyDict
import plot_trajectory, plot_surface, plot_2D
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    with open(args.path) as f:
        cfg = EasyDict(yaml.safe_load(f))
        args = plot_trajectory.plot(cfg)
        args = plot_surface.plot(cfg, args)
        plot_2D.plot(args)

