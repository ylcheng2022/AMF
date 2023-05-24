import numpy as np
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter
from data.custom_dataset import NeighborsDataset



@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)

        constrastive_features, cluster_output = model(images)
        output = memory_bank.weighted_knn(constrastive_features)

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg


"""
torch.no_grad()是一个上下文管理器，用来禁止梯度的计算，通常用在网络推断(eval)中，可以减少计算内存的使用量
被torch.no_grad()包裹起来的部分不会被追踪梯度，虽然仍可以前向传播进行计算得到输出，但计算过程(grad_fn)不会被记录，也就不能反向传播更新参数。
  具体地，对非叶子节点来说
      非叶子节点的requires_grad属性变为了False
      非叶子节点的grad_fn属性变为了None
这样便不会计算非叶子节点的梯度。因此，虽然叶子结点(模型各层的可学习参数)的requires_grad属性没有改变(依然为True)，也不会计算梯度，grad属性为None，且如果使用loss.backward()会报错(因为第一个非叶子节点(loss)的requires_grad属性为False，grad_fn属性为None)。因此，模型的可学习参数不会更新。
torch.no_grad()不会影响dropout和batchnorm层在train和eval时的行为

在PyTorch中进行validation时，会使用model.eval()切换到测试模式
该模式用于通知dropout层和batchnorm层切换至val模式
在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值(这里的学习是指在训练阶段数据前向传播的过程中累积更新的mean和var值)
关于对batchnorm层的影响的详细分析见下面的batch_normalization层部分，这里坑很多!!!
该模式不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，(代码示例一)，具体地
    叶子结点(模型各层的可学习参数)的requires_grad属性没有改变(依然为True)
    非叶子节点的requires_grad属性为True
    非叶子节点的grad_fn属性不为None
因此，该模式不会影响各层的gradient计算行为，甚至loss.backward()还能正常运行计算梯度(通常不使用)
注意，训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，相同的数据输出结果也会改变。这是model中含有BN层和Dropout所带来的的性质 (代码示例二)
如果不在意显存大小和计算时间的话，仅仅使用model.eval()已足够得到正确的validation/test的结果(在validation/test时不写loss.backward())；而with torch.no_grad()则是更进一步加速和节省gpu空间（因为不用计算和存储梯度），从而可以更快计算，也可以跑更大的batch来测试。
"""
@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False, self_labeling=False):
    # Make predictions on a dataset with neighbors
    model.eval()

    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features4_512 = torch.zeros((len(dataloader.sampler), 64)).cuda()
        features4_128 = torch.zeros((len(dataloader.sampler), 128)).cuda()
        target = torch.LongTensor(len(dataloader.sampler)).cuda()


    if isinstance(dataloader.dataset, NeighborsDataset): # Also return the neighbors
        key_ = 'anchor'
    else:
        key_ = 'image'

    ptr = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0]
        res = model(images)
        # constrastive_features, cluster_outs = model(input_)
        # cluster_outs是一个列表，列表长度为nheads。列表元素output[0]: [b, clusters]
        if return_features:
            features4_128[ptr: ptr+bs].copy_(res[0].detach())
            features4_512[ptr: ptr+bs].copy_(res[1][0].detach())
            target[ptr: ptr+bs].copy_(batch['target'].detach())
            ptr += bs

    if return_features:
        return [features4_128.cpu(), features4_512.cpu(), target.cpu()]



@torch.no_grad()
def scan_evaluate(predictions):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1,1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()

        # Total loss
        total_loss = - entropy_loss + consistency_loss

        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}
