import random
import typing
from collections import namedtuple

import torch
from torch._C import device

from codebase.torchutils.common import auto_device, compute_flops


NONE = "none"
SKIP = "skip_connect"
CONV1X1 = "nor_conv_1x1"
CONV3X3 = "nor_conv_3x3"
AVGPOOL3X3 = "avg_pool_3x3"

VALID_OPERATIONS = [NONE, SKIP, CONV1X1, CONV3X3, AVGPOOL3X3]

N_OPERATIONS = 6
AVAILABLE_OPERATIONS = list(range(len(VALID_OPERATIONS)))
# following the setting in OFA
# N_UNITS = 5
# DEPTHS = [2, 3, 4]
# N_LAYERS_PER_UNIT = max(DEPTHS)
# N_DEPTHS = len(DEPTHS)
# EXPAND_RATIOS = [3, 4, 6]
# N_EXPAND_RATIOS = len(EXPAND_RATIOS)
# KERNEL_SIZES = [3, 5, 7]
# N_KERNEL_SIZES = len(KERNEL_SIZES)
# AVAILABLE_RESOLUTIONS = [192, 208, 224, 240, 256]
# N_AVAILABLE_RESOLUTIONS = len(AVAILABLE_RESOLUTIONS)

# N_LAYERS = N_UNITS * N_LAYERS_PER_UNIT


def split(items, separator=","):
    return [int(item) for item in items.split(separator)]


def join(items, separator=","):
    return separator.join(map(str, items))


nasbench201_embeddings = torch.eye(n=N_OPERATIONS, dtype=torch.float, requires_grad=False)

T = typing.TypeVar("T")


class NASBench201Architecture:
    def __init__(self, operations):
        self.ops = self.operations = operations
        self.metadata = dict()
        # self.top1_acc = 0.0
        # self.train_accuracy = 0.0
        # self.validation_accuracy = 0.0
        # self.test_accuracy = 0.0
        # self.madds = 1000.0
        # self.latency = 0.0
        # self.prune()

        self._tensor = None

    @classmethod
    def random(cls, has_resolution=False):
        operations = random.choices(AVAILABLE_OPERATIONS, k=N_OPERATIONS)
        return cls(operations)

    @classmethod
    def from_string(cls: T, arch_str: str) -> T:
        node_strs = arch_str.split('+')
        genotypes = []
        for i, node_str in enumerate(node_strs):
            inputs = list(filter(lambda x: x != '', node_str.split('|')))
            for xinput in inputs:
                assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
            inputs = (xi.split('~') for xi in inputs)
            input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
            genotypes.append(input_infos)
        arch_ints = []
        for tmp in genotypes:
            for (op, _) in tmp:
                arch_ints.append(VALID_OPERATIONS.index(op))
        return cls.from_ints(arch_ints)

    def to_string(self) -> str:
        return f"|{VALID_OPERATIONS[self.ops[0]]}~0|" + "+" + \
            f"|{VALID_OPERATIONS[self.ops[1]]}~0|{VALID_OPERATIONS[self.ops[2]]}~1|" + "+" + \
            f"|{VALID_OPERATIONS[self.ops[3]]}~0|{VALID_OPERATIONS[self.ops[4]]}~1|{VALID_OPERATIONS[self.ops[5]]}~2|"

    @classmethod
    def from_ints(cls, arch_ints):
        return cls(arch_ints)

    def to_ints(self):
        return self.operations

    def to_tensor(self):
        if self._tensor is None:
            with torch.no_grad():
                embeddings = []
                index = torch.tensor(self.to_ints(), dtype=torch.long)
                embeddings.append(torch.index_select(nasbench201_embeddings, dim=0, index=index).flatten())
                self._tensor = torch.cat(embeddings)
        return self._tensor

    def obtain_acc_by(self, database):
        self.metadata = database.fetch_by_spec(self).metadata
        # self.test_accuracy = database.fetch_by_spec(self).test_accuracy

    def obtain_madds_by(self, database):
        return 0.0
    # def obtain_acc_by(self, acc_pred):
    #     self.top1_acc = acc_pred(self.to_tensor().unsqueeze(0).to(device=auto_device)).view([]).item()

    # def obtain_madds_by(self, supernet, resolution=224):
    #     supernet.set_active_subnet(ks=self.ks, e=self.ratios, d=self.depths)
    #     ofa_childnet = supernet.get_active_subnet(preserve_weight=False)
    #     self.madds = compute_flops(ofa_childnet, (1, 3, resolution, resolution), list(ofa_childnet.parameters())[0].device)/1e6

    def apply(self, edit):
        arch_ints = self.to_ints()
        for index, target in edit:
            arch_ints[index] = target
        return self.from_ints(arch_ints)

    @classmethod
    def from_lstm(cls, arch_seq):
        return cls.from_ints(arch_seq)

    def __hash__(self):
        return hash(self.to_string())
