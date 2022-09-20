import math


class WarmupCosineAnnealingLR(object):
    def __init__(self, optimizer, T_warmup, T_max, eta_min=0, last_epoch=-1):
        self.T_warmup = T_warmup
        self.T_max = T_max
        self.eta_min = eta_min

        self.optimizer = optimizer

        self.base_lr = optimizer.param_groups[0]["lr"]
        self.last_epoch = -1

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        if epoch < self.T_warmup:
            lr = self.base_lr * epoch / self.T_warmup
        else:
            lr = self.eta_min + (self.base_lr - self.eta_min) * \
                (1 + math.cos(math.pi * (epoch-self.T_warmup) / (self.T_max-self.T_warmup))) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
