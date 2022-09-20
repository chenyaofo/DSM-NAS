import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseController
from .gnn import EdgeGraphSAGE, NoEdgeGraph

from codebase.core.arch_representation.ofa import N_LAYERS_PER_UNIT, N_AVAILABLE_RESOLUTIONS
from codebase.core.dynamic_graph import DynamicGraph
from codebase.torchutils.common import auto_device
from codebase.torchutils import logger


class OFAModificationController(BaseController):
    def __init__(self, K=10, n_steps=2, gnn_layers=2, in_features=201, has_resolution=False,
                 n_layers=20, n_operations=10, hidden_size=64,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5,
                 batch_size=1, device=auto_device, no_edge=False):
        super(OFAModificationController, self).__init__(hidden_size, device)
        self.K = K
        self.n_steps = n_steps
        self.has_resolution = has_resolution
        self.n_layers = n_layers
        self.n_operations = n_operations
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.no_edge = no_edge

        self.batch_size = batch_size
        self.device = device

        self.gnn = EdgeGraphSAGE(gnn_layers, in_features, hidden_size, hidden_size)

        self.node_decision = nn.Linear(self.hidden_size, K)
        self.layer_decision = nn.Linear(self.hidden_size, n_layers)
        self.op_decision = nn.Linear(self.hidden_size, n_operations)

        self.emb_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.hid_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_attn = nn.Linear(self.hidden_size, 1, bias=False)

        self.node_embedding = nn.Embedding(K, self.hidden_size)
        self.layer_embedding = nn.Embedding(n_layers, self.hidden_size)
        self.op_embedding = nn.Embedding(n_operations, self.hidden_size)
        if self.has_resolution:
            self.resolution_decision = nn.Linear(self.hidden_size, N_AVAILABLE_RESOLUTIONS)
            self.resolution_embedding = nn.Embedding(n_operations, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def forward(self, graph: DynamicGraph, force_uniform=False):
        node_features = graph.node_features.to(device=self.device)
        edge_features = graph.edge_features.to(device=self.device)
        if self.no_edge:
            # print("no_edge")
            with torch.no_grad():
                edge_features=edge_features.detach()
                edge_features.requires_grad=False
                edge_features.zero_()
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
            # which layer
            if force_uniform:
                logits = torch.zeros(self.n_layers, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.layer_decision(hx).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            probs = F.softmax(logits, dim=-1)
            # logger.debug(f"Wchih Layer: {probs.tolist()}")
            action, select_log_p, entropy = self._impl(probs)
            max_entropy += math.log(probs.shape[0])

            layer_index = action.item()
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            embed = self.layer_embedding(action)
            if self.has_resolution and layer_index == 0:
                if force_uniform:
                    logits = torch.zeros(N_AVAILABLE_RESOLUTIONS, device=self.device)
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.resolution_decision(hx).view(-1)
                    logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
                probs = F.softmax(logits, dim=-1)
                action, select_log_p, entropy = self._impl(probs)
                max_entropy += math.log(probs.shape[0])

                op_index = action.item()
                logp_buf.append(select_log_p)
                entropy_buf.append(entropy)

                embed = self.resolution_embedding(action)
            else:
                if force_uniform:
                    logits = torch.zeros(self.n_operations, device=self.device)
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.op_decision(hx).view(-1)
                    logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
                is_front_split_line = 3 if self.has_resolution else 2
                if layer_index % N_LAYERS_PER_UNIT < is_front_split_line:
                    # here, the layer cannot be deleted, so we ignore the first decision (0,0)
                    logits = logits[1:]
                probs = F.softmax(logits, dim=-1)
                # logger.debug(f"Wchih Operation: {probs.tolist()}")
                action, select_log_p, entropy = self._impl(probs)
                max_entropy += math.log(probs.shape[0])

                if layer_index % N_LAYERS_PER_UNIT < is_front_split_line:
                    op_index = action.item() + 1
                else:
                    op_index = action.item()
                logp_buf.append(select_log_p)
                entropy_buf.append(entropy)

                embed = self.op_embedding(action)

            edit_seq.append((layer_index, op_index))

        return edit_seq, sum(logp_buf), sum(entropy_buf), max_entropy


class OFAController(BaseController):
    def __init__(self, n_layers=20, n_operations=10, has_resolution=False, hidden_size=64,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5,
                 batch_size=1, device=auto_device):
        super(OFAController, self).__init__(hidden_size, device)
        self.n_layers = n_layers
        self.n_operations = n_operations
        self.has_resolution = has_resolution
        self.hidden_size = hidden_size

        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce

        self.batch_size = batch_size
        self.device = device

        # self.gnn = EdgeGraphSAGE(gnn_layers, in_features, hidden_size, hidden_size)

        # self.node_decision = nn.Linear(self.hidden_size, K)
        # self.layer_decision = nn.Linear(self.hidden_size, n_layers)
        if self.has_resolution:
            self.resolution_decision = nn.Linear(self.hidden_size, N_AVAILABLE_RESOLUTIONS)
        self.op_decision = nn.Linear(self.hidden_size, n_operations)

        # self.emb_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.hid_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.v_attn = nn.Linear(self.hidden_size, 1, bias=False)

        # self.node_embedding = nn.Embedding(K, self.hidden_size)
        # self.layer_embedding = nn.Embedding(n_layers, self.hidden_size)
        if self.has_resolution:
            self.resolution_embedding = nn.Embedding(N_AVAILABLE_RESOLUTIONS, self.hidden_size)
        self.op_embedding = nn.Embedding(n_operations, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def forward(self, force_uniform=False):
        embed = self._zeros(self.batch_size)
        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)

        arch_seq = []
        logp_buf = []
        entropy_buf = []
        max_entropy = 0

        if self.has_resolution:
            if force_uniform:
                logits = torch.zeros(N_AVAILABLE_RESOLUTIONS, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.resolution_decision(hx).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            probs = F.softmax(logits, dim=-1)
            action, select_log_p, entropy = self._impl(probs)
            max_entropy += math.log(probs.shape[0])

            r = action.item()
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            embed = self.resolution_embedding(action)
            arch_seq.append(r)

        for layer_index in range(self.n_layers):
            if force_uniform:
                logits = torch.zeros(self.n_operations, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.op_decision(hx).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            if layer_index % N_LAYERS_PER_UNIT < 2:
                # here, the layer cannot be deleted, so we ignore the first decision (0,0)
                logits = logits[1:]
            probs = F.softmax(logits, dim=-1)
            action, select_log_p, entropy = self._impl(probs)
            max_entropy += math.log(probs.shape[0])

            op = action.item()
            if layer_index % N_LAYERS_PER_UNIT < 2:
                op += 1
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            embed = self.op_embedding(action)
            arch_seq.append(op)

        return arch_seq, sum(logp_buf), sum(entropy_buf), max_entropy
