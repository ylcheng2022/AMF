"""
Authors: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import mkl
mkl.get_max_threads()

import argparse
import os
import torch
import numpy as np

from utils.config_64 import create_config
from utils.common_config import get_model, get_val_dataset, get_val_dataloader, get_val_transformations, get_optimizer
from utils.evaluate_utils import get_predictions
from termcolor import colored

# 设置可用的cuda
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Parser
parser = argparse.ArgumentParser(description='Graph Contrastive Clustering')
parser.add_argument('--config_env', default='configs/env.yml', help='Config file for the environment')
parser.add_argument('--config_exp', default='configs/end2end/end2end_Cat_vs_Dog.yml',
                    help='Config file for the experiment')
args = parser.parse_args()


def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    with open(p['log_output_file'], 'w') as fw:
        fw.write(str(p) + "\n")

    # Model
    p['index_dim'] = 1
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    # 用来返回net网络中的参数的总数目
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    # 大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    # for validate
    val_dataset = get_val_dataset(p, val_transforms)  # Dataset w/o augs for knn eval
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {} val samples'.format(len(val_dataset)))

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['end2end_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['end2end_checkpoint']), 'blue'))
        checkpoint = torch.load(p['end2end_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint)
        model.cuda()
    else:
        print(colored('No checkpoint file at {}'.format(p['end2end_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    features = get_predictions(p, val_dataloader, model, return_features=True)
    # 保存特征编码
    for i in range(len(features)):
        if i == 3:
            # 保存特征标签
            print(p['features'] + "1_" + "target")
            with open(p['features'] + "1_" + "target", 'wb') as f:
                np.save(f, features[i])
        else:
            with open(p['features'] + "1_" + str(i), 'wb') as f:
                np.save(f, features[i])


if __name__ == '__main__':
    main()