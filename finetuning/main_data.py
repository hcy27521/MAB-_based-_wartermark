import os
from argparse import ArgumentParser

import data_utils
import utils
from easydict import EasyDict
import yaml



def main(cfg):
    utils.set_seed(cfg.seed)
    data_utils.main(cfg)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='path to config')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
        # if cfg.method == 'ewe':
        #     cfg.ewe = True
        #     cfg.regen_data = False
    main(cfg)