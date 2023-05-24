import numpy as np
import torch


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        # n: 数据集大小；contrastive_head_dim: 编码维度; num_classes:聚类数量， temperature:温度系数
        self.n = n
        self.dim = dim
        self.features = torch.FloatTensor(self.n, self.dim)
        #self.features = torch.FloatTensor(self.n, 512)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C),
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def laplace_transform(self, indices):
        """
        Args:
            indices: shape: [n, topk+1]
        Returns:
            各节点度的平方根
        """
        # indices.shape: [n, topk+1]
        rows, cols = indices.shape
        distance_dict = {}
        for i in range(rows):
            distance_dict[i] = set()
        for i in range(rows):
            for j in range(cols):
                value = indices[i][j]
                distance_dict[i].add(value)
                distance_dict[value].add(i)

        final_distance = {}
        for key, value in distance_dict.items():
            final_distance[key] = np.sqrt((3.0 / len(value)))
        return final_distance

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        # 挖掘每个样本的topk最近邻
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        #index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included

        # evaluate
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            # np.take() 函数用法
            # np.take(a, indices, axis=None, out=None, mode='raise')
            # 作用： 沿轴从数组中获取元素。
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)

            # for ablation study
            detail_acc = []
            #tmp_distances, tmp_indices = index.search(features, 50+1) # Sample itself is included
            #for i in range(0, 51, 5):
            #    if i == 0:
            #        tmp_neighbor_targets = np.take(targets, tmp_indices[:,1 : 2], axis=0) # Exclude sample itself for eval
            #        tmp_anchor_targets = np.repeat(targets.reshape(-1,1), 1, axis=1)
            #    else:
            #        tmp_neighbor_targets = np.take(targets, tmp_indices[:,1 : i + 1], axis=0) # Exclude sample itself for eval
            #        tmp_anchor_targets = np.repeat(targets.reshape(-1,1), i, axis=1)
            #    tmp_accuracy = np.mean(tmp_neighbor_targets == tmp_anchor_targets)
            #    print ("topk: {}, acc: {}".format(i, tmp_accuracy))
            #    detail_acc.append("topk: {}, acc: {}\n".format(i, tmp_accuracy))

            return indices, accuracy, detail_acc

        else:
            return indices

    def reset(self):
        self.ptr = 0

    def update(self, features, targets):
        b = features.size(0)

        assert(b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
