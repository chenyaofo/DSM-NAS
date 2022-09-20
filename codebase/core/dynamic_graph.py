import torch

import itertools
from functools import reduce


def mean(items):
    return sum(items)/len(items)


def _create_edge(n_nodes, perfs):
    edge_index = []
    for i, j in itertools.product(range(n_nodes), range(n_nodes)):
        if i == j:
            continue
        else:
            if perfs[i] <= perfs[j]:
                if (j, i) not in edge_index:
                    edge_index.append((i, j))
    return edge_index


class DynamicGraph:
    def __init__(self, nodes):
        self.nodes = nodes
        self._node_one_hot = None
        self._edge_ont_hot = None
        self.edge_index = None
        self.last_update_iter_ = 0
        if hasattr(self.nodes[0], "values"):
            self.perfs = [[(0, node.values, node.top1_acc)] for node in self.nodes]
        else:
            self.perfs = [[(0, node.top1_acc)] for node in self.nodes]
        self.compute_one_hot()

    def at(self, index):
        return self.nodes[index]

    def compute_one_hot(self):
        self._node_one_hot = torch.stack([node.to_tensor() for node in self.nodes], dim=0)
        accuracies = torch.tensor(self.accuracies, dtype=torch.float, device=self._node_one_hot.device).view(-1, 1)
        # print(self._node_one_hot.shape, accuracies.shape)
        self._node_one_hot = torch.cat([self._node_one_hot, accuracies], dim=1)
        self.edge_index = _create_edge(len(self.nodes), self.accuracies)

        edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=self._node_one_hot.device)
        start, end = torch.unbind(edge_index, dim=1)
        node_start_features = torch.index_select(self._node_one_hot, dim=0, index=start)
        node_end_features = torch.index_select(self._node_one_hot, dim=0, index=end)
        self._edge_ont_hot = node_end_features-node_start_features

    @property
    def node_features(self):
        return self._node_one_hot

    @property
    def edge_features(self):
        return self._edge_ont_hot

    @property
    def accuracies(self):
        return [arch.top1_acc for arch in self.nodes]

    @property
    def min_accuracy(self):
        return min(self.accuracies)

    @property
    def max_accuracy(self):
        return max(self.accuracies)

    @property
    def mean_accuracy(self):
        return mean(self.accuracies)

    def update(self, index, node, iter_, madds=None, supernet=None):
        replace = False
        if node.top1_acc > self.nodes[index].top1_acc:
            if madds is None:
                replace = True
            else:
                node.obtain_madds_by(supernet)
                if node.madds <= madds:
                    replace = True

        if replace:
            self.nodes[index] = node
            self.compute_one_hot()
            self.last_update_iter_ = iter_
            if hasattr(node, "values"):
                self.perfs[index].append((iter_, node.values, node.top1_acc))
            else:
                self.perfs[index].append((iter_, node.top1_acc))
            return 1
        else:
            return 0

    @property
    def diversity(self):
        arch_repres = [node.to_ints() for node in self.nodes]
        arch_repres = torch.tensor(arch_repres, dtype=torch.long)
        index = torch.arange(len(self.nodes), dtype=torch.long)
        index = torch.combinations(index, r=2)
        start, end = index.unbind(dim=1)
        start_archs = arch_repres.index_select(dim=0, index=start)
        end_archs = arch_repres.index_select(dim=0, index=end)
        difference = (start_archs != end_archs).float().sum(dim=1)
        diversity_mean = difference.mean().item()
        diversity_std = difference.std().item()
        return diversity_mean, diversity_std

    @property
    def min_architecture(self):
        index, perf = -1, 1
        arch = None
        for i, node in enumerate(self.nodes):
            if node.top1_acc < perf:
                perf = node.top1_acc
                index = i
                arch = node
        return arch

    @property
    def max_architecture(self):
        index, perf = -1, -1
        arch = None
        for i, node in enumerate(self.nodes):
            if node.top1_acc > perf:
                perf = node.top1_acc
                index = i
                arch = node
        return arch

    @property
    def min_architecture_index(self):
        index, perf = -1, 1
        for i, node in enumerate(self.nodes):
            if node.top1_acc < perf:
                perf = node.top1_acc
                index = i
        return index

    @property
    def max_architecture_index(self):
        index, perf = -1, -1
        for i, node in enumerate(self.nodes):
            if node.top1_acc > perf:
                perf = node.top1_acc
                index = i
        return index
