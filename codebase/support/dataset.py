import json
import torch
from torch.utils.data import Dataset

from codebase.core.arch_representation.ofa import OFAArchitecture


class OFAArchitectureDataset(Dataset):
    def __init__(self, path, train, seed, proportion, width, has_resolution=False):
        self.has_resolution = has_resolution
        with open(path, "r") as f:
            self.archs = json.load(f)
        n_total = len(self.archs)
        g = torch.Generator()
        g.manual_seed(seed)
        index = torch.randperm(n_total, generator=g).tolist()
        n_samples = int(n_total * proportion)
        if train:
            index = index[:n_samples]
        else:
            index = index[n_samples:]
        split = [self.archs[i] for i in index]

        self.arch_tensors = []
        self.top1_accs = []

        for arch in split:
            arch_string = arch["arch"]
            ofa_arch = OFAArchitecture.from_string(arch_string)
            if self.has_resolution:
                ofa_arch.resolution = arch["w"+str(width)]["resolution"]
            self.arch_tensors.append(ofa_arch.to_tensor())
            self.top1_accs.append(arch["w"+str(width)]["top1_acc"]*100)

    def __getitem__(self, index):
        return self.arch_tensors[index], self.top1_accs[index]

    def __len__(self):
        return len(self.arch_tensors)
