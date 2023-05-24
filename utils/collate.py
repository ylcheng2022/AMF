import torch
import numpy as np
import collections
# torch1.10报错，更换int_classes， string_classes
# from torch._six import string_classes, int_classes
int_classes = int
string_classes = str

""" Custom collate function """
# 即用于collate的function，用于整理数据的函数。
# batch[index] = {
# 'augmented': augmented, torch.Tensor
# 'target': target, np.int64
# 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}
# 'image': image, torch.Tensor
# 'neighbor': neighbor, torch.Tensor
# }
def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))
