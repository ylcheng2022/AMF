import os
import torch
import numpy as np
import errno

# 判断是否存在directory文件夹，如果不存在则创建
def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


#     progress = ProgressMeter(len(train_loader),
#         [losses, constrastive_losses, cluster_losses, consistency_losses, entropy_losses],
#         prefix="Epoch: [{}]".format(epoch), output_file=log_output_file)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", output_file=None):
        # num_batches = n / batch_size,训练批次的总数量
        # meters = [losses, constrastive_losses, cluster_losses, consistency_losses, entropy_losses]，5类损失的AverageMeter对象
        # prefix：输出前缀
        # output_file： 输出文件路径
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.output_file = output_file
        self.fw = open(self.output_file, 'a+')

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        self.fw.write('\t'.join(entries) + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output, _ = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))

@torch.no_grad()
def fill_memory_bank_mean(loader, aug_feat_memory, org_feat_memory, memory_bank):
    memory_bank.reset()
    # batch = {
    # 'augmented': augmented, torch.stack(batch, 0)
    # 'target': target, torch.LongTensor(batch)
    # 'meta': {'im_size': img_size, [torch.LongTensor(batch), torch.LongTensor(batch)]
    #           'index': index, torch.LongTensor(batch)
    #           'class_name': class_name, str_list}
    # 'image': image, torch.stack(batch, 0)
    # 'neighbor': neighbor, torch.stack(batch, 0)
    # }
    for i, batch in enumerate(loader):
        #images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = []
        indexes = batch['meta']['index']
        for index in indexes:
            key = index.item()
            aug_feat = aug_feat_memory.pop(key)
            # aug_feat是一个4个元素的列表，列表里面的每个元素是最新的一轮的低维编码（对比学习所使用到的编码），维度为[1, ndim]
            if len(aug_feat) > 1:
                aug_feat = torch.cat(aug_feat, 0)
                mean_aug_feat = torch.mean(aug_feat, 0, True)
            else:
                mean_aug_feat = aug_feat[0]
            output.append(mean_aug_feat)
            #output.append(aug_feat)

        output = torch.cat(output, 0)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


def confusion_matrix(predictions, gt, class_names, output_file=None):
    """
    Args:
        predictions: reordered_preds
        gt: targets
        class_names: 对应的类别的名称列表【0-9】
        output_file: 输出文件路径
    Returns:
        None
    """
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, dim=1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)
    """
        # np.ndenumerate()效果等同与enumerate，并且支持对多维数据的输出：
        # np.nindex(*shape) 用于求数列中元素的下标
    """
    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' %(100*z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass
    """ 
        问题1： 在绘制多个图形时，经常会看到一些子图的标签在它们的相邻子图上重叠
        解决方法: 在plt.show()前加上：plt.tight_layout() 
        问题2： python绘图保存的图像坐标轴显示不全以及图片周围空白较大的问题
        解决方法: 只需要在plt.savefig()函数中，加入一个参数。即plt.savefig(“path”, bbox_inches=‘tight’)
    """
    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
