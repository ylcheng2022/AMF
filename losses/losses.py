import torch
import math
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-8


class RGCLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(RGCLoss, self).__init__()
        self.temperature = temperature


    def forward(self, features, neighbor_topk_features, neighbor_weights, neighbor_num):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
            - neighbor_topk_features.shape: [3b, 128]
            - neighbor_weights.shape: [b, 3b]
        output:
            - loss: loss computed according to SimCLR
        """
        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        # torch.unbind()作用：对某一个维度进行长度为1的切片，并将所有切片结果返回。
        # torch.unbind(features, dim=1) 返回（features[b, dim], features[b, dim]）
        # torch.cat()是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。
        # torch.cat(torch.unbind(features, dim=1), dim=0) 返回 Tensor.size() = [2b, dim]
        if neighbor_topk_features is not None:
            contrast_features = torch.cat([contrast_features, neighbor_topk_features], dim=0).cuda()
            # contrast_features.shape: [5b, dim]

        anchor = features[:, 0]
        # anchor也可以用通过torch.unbind(features, dim=1)[0] 获取
        # anchor.shape: [b, dim]

        # Dot product
        # contrast_features.size(): [2b, dim] 。前b行为自己的特征，后b行为增广的特征
        # dot_product.size(): [b, 2b]
        # or
        # anchor.size(): [b, dim]
        # contrast_features.size(): [5b, dim] 。前b行为自己的特征，后b-2b行为增广的特征,2b-3b行为1近邻节点特征，3b-4b行为2近邻节点特征，后4b-5b为3近邻节点特征.
        # # dot_product.size(): [b, 5b]
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability 数值稳定性的对数和技巧
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        if neighbor_topk_features is not None:
            mask = torch.cat([torch.zeros([b, b]).cuda(), mask.repeat(1, (neighbor_num + 1))], dim=1)
        #     mask.size(): [b, 5b]
        else:
            mask = torch.cat([torch.zeros([b, b]).cuda(), mask], dim=1)
        #     mask.size(): [b, 2b]

        # Log-softmax
        exp_logits = torch.exp(logits)

        if neighbor_weights is not None:
            weights = torch.cat([torch.ones([b, b]).cuda(), torch.ones([b, b]).cuda(), neighbor_weights], dim=1)
            # weights.shape: [b, 5b]
            # exp_logits.shape: [b, 5b]
            exp_logits = exp_logits * weights

        if neighbor_topk_features is not None:
            ones_matrix = torch.ones_like(mask)
            prob = exp_logits / ((ones_matrix - mask).cuda() * exp_logits).sum(1, keepdim=True)
        else:
            # 两句的区别在于，分母是否排除正例的影响
            # 排除正例的影响
            prob = exp_logits / ( (torch.ones_like(mask) - mask).cuda() * exp_logits).sum(1, keepdim=True)
            # 不排除正例的影响
            # prob = exp_logits / (exp_logits).sum(1, keepdim=True)

        # Mean log-likelihood for positive
        loss = - (torch.log((mask * prob).sum(1))).mean()

        return loss
