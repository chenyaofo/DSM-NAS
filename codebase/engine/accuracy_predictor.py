import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from codebase.support.mlp import MultilayerPerceptron
from codebase.support.dataset import OFAArchitectureDataset

from codebase.torchutils import logger
from codebase.torchutils.metrics import AverageMetric, GroupMetric
from codebase.torchutils.common import auto_device
from codebase.torchutils import logger


def _impl(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer = None,
          scheduler: optim.lr_scheduler._LRScheduler = None, huber_beta=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_avg = GroupMetric(2, AverageMetric)
    with torch.set_grad_enabled(is_train):
        for archs, accs in loader:
            accs = accs.float()
            archs, accs = archs.to(device=auto_device), accs.to(device=auto_device)
            preds = model(archs)
            preds.squeeze_(dim=1)
            with torch.no_grad():
                l1loss = F.l1_loss(preds, accs)
            if huber_beta is None:
                loss = F.mse_loss(preds, accs)
            else:
                loss = nn.SmoothL1Loss(beta=huber_beta)(preds, accs)
            loss_avg.update([l1loss, loss])
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return loss_avg.compute()


def train_acc_pred(max_epochs, model, train_loader, val_loader, optimizer, scheduler, log_interval, huber_beta):
    logger.info(f"huber_beta={huber_beta}")
    for epoch in range(1, max_epochs+1):
        _impl(model, train_loader, optimizer, scheduler, huber_beta=huber_beta)
        if epoch % log_interval == 0:
            l1loss, *_ = _impl(model, val_loader)
            logger.info(f"Train accuracy predictor, iter={epoch:04d}/{max_epochs:04d}, "
                        f"L1 Loss={l1loss:.6f}, diff={l1loss:.4f}%.")


def train_acc_pred_complete(max_epochs=200, data_root="assets/ofa/ofa_data.json", use_resolution=False,
                            seed=2020, training_proportion=0.9, width="1.2",
                            batch_size=256, num_workers=1,
                            num_layers=3, in_features=10*20, hidden_features=128, out_features=1,
                            lr=0.1, use_our_supernet=False, drop_rate=0.5, huber_beta=None):
    in_features = 20*10+5 if use_resolution else 20*10
    # if use_our_supernet:
    #     data_root = "assets/ofa/our_data.json"
    # else:
    #     if use_resolution:
    #         data_root = "assets/ofa/ofa_dr_data.json"
    #     else:
    #         data_root = "assets/ofa/ofa_data.json"
    # data_root = "assets/ofa/ofa_dr_data.json" if use_resolution else "assets/ofa/our_data.json"
    logger.info(f"drop_rate={drop_rate}")
    acc_predictor = MultilayerPerceptron(num_layers, in_features, hidden_features, out_features,
                                         drop_rate=drop_rate)
    # if os.path.exists("assets/ofa/accuracy_predictor.pt") and not use_resolution:
    acc_predictor.load_state_dict(torch.load("assets/accuracy_predictor.pt", map_location="cpu"))
    #     acc_predictor.to(device=auto_device)
    # elif os.path.exists("assets/ofa/r_accuracy_predictor.pt") and use_resolution:
    #     acc_predictor.load_state_dict(torch.load("assets/ofa/r_accuracy_predictor.pt", map_location="cpu"))
    #     acc_predictor.to(device=auto_device)
    # else:
    acc_predictor.to(device=auto_device)
    return acc_predictor
    trainset = OFAArchitectureDataset(data_root, True, seed, training_proportion, width, has_resolution=use_resolution)
    valset = OFAArchitectureDataset(data_root, False, seed, training_proportion, width, has_resolution=use_resolution)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = optim.SGD(acc_predictor.parameters(), lr=lr,
                          weight_decay=3e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=0)
    train_acc_pred(max_epochs, acc_predictor, train_loader, val_loader, optimizer, scheduler,
                   log_interval=max_epochs//10, huber_beta=huber_beta)
    return acc_predictor
