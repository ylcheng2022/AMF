import argparse
import os
import torch

from utils.config_64 import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset, \
    get_train_dataloader, get_train_transformations, \
    get_val_transformations, get_optimizer, \
    adjust_learning_rate
from utils.train_utils import gcc_train
from termcolor import colored
from utils.aug_feat import AugFeat

# 设置可用的cuda
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# Parser
parser = argparse.ArgumentParser(description='Graph Contrastive Clustering')
parser.add_argument('--config_env', default='configs/env.yml', help='Config file for the environment')
parser.add_argument('--config_exp', default='configs/end2end/end2end_cifar10.yml',
                    help='Config file for the experiment')
args = parser.parse_args()


def main():
    org_feat_memory = AugFeat('./org_feat_memory', 4)
    # 用于记录增广特征的编码，最多纪录4轮
    aug_feat_memory = AugFeat('./aug_feat_memory', 4)

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    with open(p['log_output_file'], 'w') as fw:
        fw.write(str(p) + "\n")

    # Model
    print(colored('Retrieve model', 'blue'))
    p['index_dim'] = 1
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
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    # for train
    train_dataset = get_train_dataset(p, train_transforms, to_end2end_dataset=True,
                                      split='train+unlabeled')
    train_dataloader = get_train_dataloader(p, train_dataset)

    print('Dataset contains {} samples'.format(len(train_dataset)))
    print(colored('Build MemoryBank', 'blue'))

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion1 is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['end2end_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['end2end_checkpoint']), 'blue'))
        checkpoint = torch.load(p['end2end_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        # start_epoch = checkpoint['epoch'] - 1   # 10000 for evaluate directly
        start_epoch = checkpoint['epoch'] + 1000  # 10000 for evaluate directly
    else:
        print(colored('No checkpoint file at {}'.format(p['end2end_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        if epoch <= 500:
            print('Train pretext...')
            gcc_train(train_dataloader, model, criterion, None, optimizer,
                      epoch, aug_feat_memory, org_feat_memory, p['log_output_file'], True)
        else:
            break

        if epoch > 0 and epoch % 5 == 0:
            torch.save(model.state_dict(), p['end2end_model'])


if __name__ == '__main__':
    main()
