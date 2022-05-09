import os
import random

import numpy as np
import torch


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    elif not os.path.exists(paths):
        os.makedirs(paths)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_save_path(config):
    save_path = os.path.join(config.save_root, config.dataset)

    if config.mode == 'train':
        # train mode needs path for save parameter
        save_path = os.path.join(save_path, 'param')
    else:
        # test mode needs path for save image
        save_path = os.path.join(save_path, 'img')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('make directory ...', str(save_path))

    return save_path