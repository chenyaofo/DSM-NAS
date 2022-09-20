import json
import typing

from codebase.core.arch_representation.nasbench201 import NASBench201Architecture


T = typing.TypeVar("T")


class NASBench201DataBase:
    def __init__(self):
        self.archs = {}
        self.items = []
        self.dataset = None

    @classmethod
    def from_file(cls: T, path, dataset) -> T:
        with open(path, "r") as f:
            raw_archs = json.load(f)
        overall_archs = {}
        for dataset_, archs in raw_archs.items():
            for arch in archs:
                if arch["arch"] in overall_archs:
                    a = overall_archs[arch["arch"]]
                else:
                    a = NASBench201Architecture.from_string(arch["arch"])
                a.metadata[dataset_] = arch
                if not arch["arch"] in overall_archs:
                    overall_archs[a.to_string()] = a
        # for raw_arch in raw_archs:
        #     arch = NASBench201Architecture.from_string(raw_arch["arch"])
        #     arch.train_accuracy = raw_arch["train_acc"]
        #     arch.validation_accuracy = raw_arch["val_acc"]
        #     arch.test_accuracy = raw_arch["test_acc"]
        #     archs[arch.to_string()] = arch
        database = cls()
        database.archs = overall_archs
        database.dataset = dataset
        database.items = [v for k, v in overall_archs.items()]
        database._sort()
        return database

    def _sort(self):
        sorted_items = []
        for hash_, arch in self.archs.items():
            sorted_items.append((hash_, arch.metadata[self.dataset]['test_acc']))
        sorted_items = sorted(sorted_items, key=lambda item: item[1], reverse=True)
        for i, (hash_, _) in enumerate(sorted_items, start=1):
            self.archs[hash_].metadata[self.dataset]["rank"] = i

    def fetch_by_spec(self, arch: NASBench201Architecture):
        return self.archs[arch.to_string()]

    @property
    def size(self):
        return len(self.items)
    
    def __len__(self):
        return len(self.items)