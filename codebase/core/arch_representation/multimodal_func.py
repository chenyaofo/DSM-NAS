import copy
import random
import torch

from codebase.torchutils.common import auto_device


class MultimodalFunctionPoint:
    def __init__(self, values, range_, delta):
        self.values = values
        self.range_ = range_
        self.delta = delta
        self.top1_acc = 0.0

        self._tensor = None

    @classmethod
    def random(cls, range_, delta=0.01):
        values = []
        for r in range_:
            lower_bound, upper_bound = r
            values.append(random.random() * (upper_bound-lower_bound) + lower_bound)

        return cls(values, range_, delta)

    # @classmethod
    # def from_string(cls, arch_string):
    #     '''
    #     arch string example:
    #     3,4,4,3,3:5,7,5,0,7,7,7,3,3,5,3,7,7,5,5,0,7,7,5,0:4,3,4,0,6,6,6,6,4,6,6,6,3,4,3,0,6,3,3,0
    #     '''
    #     operations, connections = arch_string.split(":")
    #     return cls(split(operations), split(connections))

    def __str__(self):
        string = []
        for v in self.values:
            string.append(f"{v:.4f}")
        return "("+",".join(string)+")"

    # def to_ints(self):
    #     return self.operations + self.connections

    # @classmethod
    # def from_ints(cls, arch_ints):
    #     n = len(arch_ints)
    #     return cls(arch_ints[:n], arch_ints[n:])

    def to_tensor(self):
        if self._tensor is None:
            with torch.no_grad():
                self._tensor = torch.tensor(self.values)
        return self._tensor

    def obtain_acc_by(self, acc_pred):
        self.top1_acc = acc_pred(*self.values)

    def apply(self, edit):
        values = copy.deepcopy(self.values)
        # print(edit)
        for _, movement in enumerate(edit):
            for i, up_down in enumerate(movement):
                values[i] += (1 if up_down == 1 else -1) * self.delta
        return MultimodalFunctionPoint(values, self.range_, self.delta)

    @classmethod
    def from_lstm(cls, seq):
        values = []
        for item in seq:
            value = 0
            for i, digit in enumerate(item[1:]):
                value += digit * 10**(-i)
            value *= 1 if item[0] == 1 else -1
            values.append(value)
        return cls(values, None, None)

    def __hash__(self):
        return hash(self.to_string())
