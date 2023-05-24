import torch
from utils.utils import AverageMeter, ProgressMeter
import torch.nn.functional as F

def gcc_train(train_loader, model, criterion1, criterion2, optimizer, epoch, aug_feat_memory, org_feat_memory, log_output_file, only_train_pretext=True):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    constrastive_losses = AverageMeter('Constrast Loss', ':.4e')
    cluster_losses = AverageMeter('Cluster Loss', ':.4e')
    consistency_losses = AverageMeter('Consist Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses, constrastive_losses, cluster_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch), output_file=log_output_file)

    model.train()

    for i, batch in enumerate(train_loader):
        if only_train_pretext:
            # non_blocking=True 非堵塞，异步操作。 具体来说， 在Host上是 non-blocking的，也就是说数据传输kernel一启动，控制权就直接回到Host上了，即Host不需要等数据从Host传输到Device了。
            # 注意non_blocking=True后面紧跟与之相关的语句时，就会需要做同步操作，等到data transfer完成为止，如下面代码示例 x=x.cuda(non_blocking=True) y = model(x)
            # Pytorch官方的建议是pin_memory=True和non_blocking=True搭配使用，这样能使得data transfer可以overlap computation。
            images = batch['image'].cuda(non_blocking=True)
            images_augmented = batch['augmented'].cuda(non_blocking=True)
        else:
            images = batch['image'].cuda(non_blocking=True)
            images_augmented = batch['augmented'].cuda(non_blocking=True)
            # 计算邻居节点的前向传播特征、
            neighbor_top1 = batch['neighbor_top1'].cuda(non_blocking=True)
            neighbor_top2 = batch['neighbor_top2'].cuda(non_blocking=True)
            neighbor_top3 = batch['neighbor_top3'].cuda(non_blocking=True)
            neighbor_top1_features, neighbor_top1_cluster_outs, _ = model(neighbor_top1)
            neighbor_top2_features, neighbor_top2_cluster_outs, _ = model(neighbor_top2)
            neighbor_top3_features, neighbor_top3_cluster_outs, _ = model(neighbor_top3)
            neighbor_top1_features = neighbor_top1_features * batch['neighbor_top1_weight'].unsqueeze(-1).cuda()
            neighbor_top2_features = neighbor_top2_features * batch['neighbor_top2_weight'].unsqueeze(-1).cuda()
            neighbor_top3_features = neighbor_top3_features * batch['neighbor_top3_weight'].unsqueeze(-1).cuda()
            b = batch['neighbor_top1_weight'].shape[0]
            fill_one_diag_zero = torch.ones([b, b]).fill_diagonal_(0).cuda()
            neighbor_weights = torch.cat([fill_one_diag_zero + torch.diag(batch['neighbor_top1_weight'].cuda()),
                                          fill_one_diag_zero + torch.diag(batch['neighbor_top2_weight'].cuda()),
                                          fill_one_diag_zero + torch.diag(batch['neighbor_top3_weight'].cuda())], dim=1)
        #     neighbor_weights.shape: [b, 3b]

        neighbors = batch['neighbor'].cuda(non_blocking=True)
        #     neighbors.shape: [b, n-dim]

        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        # input_.size(): b, 2, c, h, w
        input_ = input_.view(-1, c, h, w)
        # input_.size(): 2b, c, h, w
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        constrastive_features, _ = model(input_)
        constrastive_features = F.normalize(constrastive_features, dim=-1)
        # constrastive_features.size(): 2b, 128
        # cluster_outs.size(): 2b, 10
        constrastive_features = constrastive_features.view(b, 2, -1)
        # constrastive_features2 = constrastive_features2.view(b, 2, -1)
        # constrastive_features3 = constrastive_features3.view(b, 2, -1)
        # constrastive_features4 = constrastive_features4.view(b, 2, -1)
        # constrastive_features.size(): b, 2, 128


        constrastive_loss = criterion1(constrastive_features, None, None, 0)
        # constrastive_loss2 = criterion1(constrastive_features2, None, None, 0)
        # constrastive_loss3 = criterion1(constrastive_features3, None, None, 0)
        # constrastive_loss4 = criterion1(constrastive_features4, None, None, 0)

        # # 更新aug_feat_memory
        # aug_feat_memory.push(constrastive_features.clone().detach()[:, 1], batch['meta']['index'])

        # cluster_loss = torch.tensor([0.0]).cuda()
        # 加权获取损失
        loss = constrastive_loss

        # 记录losses constrastive_losses cluster_losses的值，传入给对应的AverageMeter对象记录
        losses.update(loss.item())
        constrastive_losses.update(loss.item())
        # cluster_losses.update(cluster_loss.item())

        # 梯度清零并损失反向传播，优化器更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每隔25个batch，进行输出打印当前batch的损失
        if i % 5 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None, output_file=None):
    """
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch), output_file=output_file)
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        if i % 25 == 0:
            progress.display(i)


def gcc_train_128(train_loader, model, criterion1, criterion2, optimizer, epoch, aug_feat_memory, org_feat_memory, log_output_file, only_train_pretext=True):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    constrastive_losses = AverageMeter('Constrast Loss', ':.4e')
    cluster_losses = AverageMeter('Cluster Loss', ':.4e')
    consistency_losses = AverageMeter('Consist Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses, constrastive_losses, cluster_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch), output_file=log_output_file)

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['augmented'].cuda(non_blocking=True)

        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        # input_.size(): b, 2, c, h, w
        input_ = input_.view(-1, c, h, w)
        # input_.size(): 2b, c, h, w
        input_ = input_.cuda(non_blocking=True)
        constrastive_features, _ = model(input_)
        constrastive_features = F.normalize(constrastive_features, dim=-1)
        # constrastive_features.size(): 2b, 128
        # cluster_outs.size(): 2b, 10
        constrastive_features = constrastive_features.view(b, 2, -1)

        constrastive_loss = criterion1(constrastive_features, None, None, 0)

        loss = constrastive_loss

        # 记录losses constrastive_losses cluster_losses的值，传入给对应的AverageMeter对象记录
        losses.update(loss.item())
        constrastive_losses.update(loss.item())
        # cluster_losses.update(cluster_loss.item())

        # 梯度清零并损失反向传播，优化器更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每隔25个batch，进行输出打印当前batch的损失
        if i % 25 == 0:
            progress.display(i)