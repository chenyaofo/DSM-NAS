import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseController
from .gnn import EdgeGraphSAGE

from codebase.core.arch_representation.ofa import N_LAYERS_PER_UNIT
from codebase.core.dynamic_graph import DynamicGraph
from codebase.torchutils.common import auto_device
from codebase.torchutils import logger

N_DIRECTIONS = 2  # increase or decrease


class MultimodalModificationController(BaseController):
    def __init__(self, K=10, n_steps=2, gnn_layers=2, in_features=2+1,
                 hidden_size=16,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5,
                 batch_size=1, device=auto_device):
        super(MultimodalModificationController, self).__init__(hidden_size, device)
        self.K = K
        self.n_steps = n_steps

        self.in_features = in_features

        self.hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce

        self.batch_size = batch_size
        self.device = device

        self.gnn = EdgeGraphSAGE(gnn_layers, in_features, hidden_size, hidden_size)

        self.node_decision = nn.Linear(self.hidden_size, K)
        self.up_or_down_decision = nn.Linear(self.hidden_size, N_DIRECTIONS)
        # self.op_decision = nn.Linear(self.hidden_size, n_operations)

        self.emb_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.hid_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_attn = nn.Linear(self.hidden_size, 1, bias=False)

        self.node_embedding = nn.Embedding(K, self.hidden_size)
        self.up_or_down_embedding = nn.Embedding(N_DIRECTIONS, self.hidden_size)
        # self.op_embedding = nn.Embedding(n_operations, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def forward(self, graph: DynamicGraph, force_uniform=False):
        node_features = graph.node_features.to(device=self.device)
        edge_features = graph.edge_features.to(device=self.device)
        graph_features, _ = self.gnn(node_features, edge_features, graph.edge_index)

        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)

        edit_seq = []
        logp_buf = []
        entropy_buf = []
        max_entropy = 0

        # decide which node is most potential
        embed = graph_features.mean(dim=0, keepdim=True)
        if force_uniform:
            logits = torch.zeros(self.K, device=self.device)
        else:
            hx, cx = self.lstm(embed, hidden)
            hidden = (hx, cx)
            query = torch.tanh(self.emb_attn(graph_features) + self.hid_attn(hx))
            logits = self.v_attn(query).view(-1)  # (n_nodes,)
            logits = self._scale_attention(logits, self.temperature, self.tanh_constant)
        probs = F.softmax(logits, dim=-1)
        # logger.debug(f"Most Potential Node: {probs.tolist()}")
        action, select_log_p, entropy = self._impl(probs)
        max_entropy += math.log(probs.shape[0])

        edit_seq.append(action.item())
        logp_buf.append(select_log_p)
        entropy_buf.append(entropy)

        embed = self.node_embedding(action)
        for step in range(self.n_steps):
            up_downs = []
            for demension in range(self.in_features-1):

                if force_uniform:
                    logits = torch.zeros(N_DIRECTIONS, device=self.device)
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.up_or_down_decision(hx).view(-1)
                    logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
                probs = F.softmax(logits, dim=-1)
                action, select_log_p, entropy = self._impl(probs)
                max_entropy += math.log(probs.shape[0])

                up_down = action.item()
                logp_buf.append(select_log_p)
                entropy_buf.append(entropy)

                embed = self.up_or_down_embedding(action)

                up_downs.append(up_down)

            edit_seq.append(up_downs)
        return edit_seq, sum(logp_buf), sum(entropy_buf), max_entropy


N_POSITIVE_NEGATIVE = 2


class MultimodalController(BaseController):
    def __init__(self, n_dimensions=2, n_dicisions=4,
                 n_can_dicisions=10,  hidden_size=16,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5,
                 batch_size=1, device=auto_device):
        super(MultimodalController, self).__init__(hidden_size, device)
        self.n_dimensions = n_dimensions
        self.n_dicisions = n_dicisions
        self.n_can_dicisions = n_can_dicisions

        # self.hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce

        self.batch_size = batch_size
        # self.device = device

        self.pn_decision = nn.Linear(self.hidden_size, N_POSITIVE_NEGATIVE)
        self.decision = nn.Linear(self.hidden_size, self.n_can_dicisions)
        # self.up_or_down_decision = nn.Linear(self.hidden_size, N_DIRECTIONS)
        # self.op_decision = nn.Linear(self.hidden_size, n_operations)

        self.emb_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.hid_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_attn = nn.Linear(self.hidden_size, 1, bias=False)

        self.pn_embedding = nn.Embedding(N_POSITIVE_NEGATIVE, self.hidden_size)
        self.embedding = nn.Embedding(self.n_can_dicisions, self.hidden_size)
        # self.up_or_down_embedding = nn.Embedding(N_DIRECTIONS, self.hidden_size)
        # self.op_embedding = nn.Embedding(n_operations, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def forward(self, force_uniform=False):
        embed = self._zeros(self.batch_size)
        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)

        arch_seq = []
        logp_buf = []
        entropy_buf = []
        max_entropy = 0

        for dimension in range(self.n_dimensions):
            number = []
            if force_uniform:
                logits = torch.zeros(N_POSITIVE_NEGATIVE, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.pn_decision(hx).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            probs = F.softmax(logits, dim=-1)
            action, select_log_p, entropy = self._impl(probs)
            max_entropy += math.log(probs.shape[0])

            number.append(action.item())
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            embed = self.pn_embedding(action)

            for disicion in range(self.n_dicisions):
                if force_uniform:
                    logits = torch.zeros(self.n_can_dicisions, device=self.device)
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.decision(hx).view(-1)
                    logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
                probs = F.softmax(logits, dim=-1)
                action, select_log_p, entropy = self._impl(probs)
                max_entropy += math.log(probs.shape[0])

                number.append(action.item())
                logp_buf.append(select_log_p)
                entropy_buf.append(entropy)

                embed = self.embedding(action)
            arch_seq.append(number)

        return arch_seq, sum(logp_buf), sum(entropy_buf), max_entropy
