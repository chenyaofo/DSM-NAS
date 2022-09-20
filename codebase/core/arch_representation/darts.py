import random
from collections import namedtuple

import torch
from torch._C import device

# from codebase.third_party.darts.genotypes import DARTS_SPACE
from codebase.torchutils.common import auto_device

# following the setting in DARTS
DARTS_SPACE = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'max_pool_3x3',
    'avg_pool_3x3',
    'none',
    'skip_connect',
    'dil_conv_3x3',
    'dil_conv_5x5'
]
N_CANDIDATE_OP = len(DARTS_SPACE)
N_NODES = 4
N_CONNECTIONS = N_OP = 8
CANDIDATE_OP_INDEXES = list(range(N_CANDIDATE_OP))


def split(items, separator=","):
    return [int(item) for item in items.split(separator)]


def join(items, separator=","):
    return separator.join(map(str, items))


darts_one_hot_embeddings = torch.eye(n=len(max(N_CANDIDATE_OP, N_CONNECTIONS)), dtype=torch.float, requires_grad=False)


class DARTSArchitecture:
    def __init__(self, operations, connections):
        self.operations = operations
        self.connections = connections
        self.top1_acc = 0.0
        self.latency = 0.0

        self._tensor = None

    @classmethod
    def random(cls):
        operations = random.choices(CANDIDATE_OP_INDEXES, k=N_OP)
        connections = [random.choice(range(i//2+2)) for i in range(N_CONNECTIONS)]
        return cls(operations, connections)

    @classmethod
    def from_string(cls, arch_string):
        '''
        arch string example:
        3,4,4,3,3:5,7,5,0,7,7,7,3,3,5,3,7,7,5,5,0,7,7,5,0:4,3,4,0,6,6,6,6,4,6,6,6,3,4,3,0,6,3,3,0
        '''
        operations, connections = arch_string.split(":")
        return cls(split(operations), split(connections))

    def to_string(self):
        '''
        arch string example:
        3,4,4,3,3:5,7,5,0,7,7,7,3,3,5,3,7,7,5,5,0,7,7,5,0:4,3,4,0,6,6,6,6,4,6,6,6,3,4,3,0,6,3,3,0
        '''
        return f"{join(self.operations)}:{join(self.connections)}"

    def to_ints(self):
        return self.operations + self.connections

    @classmethod
    def from_ints(cls, arch_ints):
        n = len(arch_ints)
        return cls(arch_ints[:n], arch_ints[n:])

    def to_tensor(self):
        if self._tensor is None:
            with torch.no_grad():
                embedding_indexes = torch.tensor(self.to_ints(), dtype=torch.long)
                embeddings = torch.index_select(darts_one_hot_embeddings, dim=0, index=embedding_indexes).flatten()
                self._tensor = embeddings
        return self._tensor

    def obtain_acc_by(self, acc_pred):
        self.top1_acc = acc_pred(self.to_tensor().unsqueeze(0).to(device=auto_device)).view([]).item()

    def apply(self, edit):
        arch_ints = self.to_ints()
        for index, target in edit:
            arch_ints[index] = target
        return self.from_ints(arch_ints)

    def __hash__(self):
        return hash(self.to_string())
