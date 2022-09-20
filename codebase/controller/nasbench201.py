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

class NASBench201Controller(BaseController):
    def __init__(self, K=10,
                 n_layers=6, n_operations=5, hidden_size=64,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5,
                 batch_size=1, device=auto_device,):
        super(NASBench201Controller, self).__init__(hidden_size, device)
        self.n_layers = n_layers
        self.n_operations = n_operations
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce

        self.batch_size = batch_size
        self.device = device

        self.op_decision = nn.Linear(self.hidden_size, n_operations)

        self.op_embedding = nn.Embedding(n_operations, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def forward(self, force_uniform=False):

        inputs = self._zeros(self.batch_size)
        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)
        embed = None

        edit_seq = []
        logp_buf = []
        entropy_buf = []
        max_entropy = 0

        # decide which node is most potential
        for i in range(self.n_layers):
            embed = inputs if embed is None else self.op_embedding(inputs)
            if force_uniform:
                logits = torch.zeros(self.n_operations, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.op_decision(hx).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            probs = F.softmax(logits, dim=-1)
            # logger.debug(f"Most Potential Node: {probs.tolist()}")
            action, select_log_p, entropy = self._impl(probs)
            # print(action.shape)
            max_entropy += math.log(probs.shape[0])

            edit_seq.append(action.item())
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            inputs = action

        return edit_seq, sum(logp_buf), sum(entropy_buf), max_entropy

class NASBench201ModificationController(BaseController):
    def __init__(self, K=10, n_steps=2, gnn_layers=2, in_features=201,
                 n_layers=6, n_operations=5, hidden_size=64,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5,
                 batch_size=1, device=auto_device, no_edge=False):
        super(NASBench201ModificationController, self).__init__(hidden_size, device)
        self.K = K
        self.n_steps = n_steps
        # self.has_resolution = has_resolution
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
        # if self.has_resolution:
        #     self.resolution_decision = nn.Linear(self.hidden_size, N_AVAILABLE_RESOLUTIONS)
        #     self.resolution_embedding = nn.Embedding(n_operations, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def forward(self, graph: DynamicGraph, force_uniform=False):
        node_features = graph.node_features.to(device=self.device)
        edge_features = graph.edge_features.to(device=self.device)
        if self.no_edge:
            # print("no_edge")
            with torch.no_grad():
                edge_features = edge_features.detach()
                edge_features.requires_grad = False
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
        # print(action.shape)
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
            if force_uniform:
                logits = torch.zeros(self.n_operations, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.op_decision(hx).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            probs = F.softmax(logits, dim=-1)
            # logger.debug(f"Wchih Operation: {probs.tolist()}")
            action, select_log_p, entropy = self._impl(probs)
            max_entropy += math.log(probs.shape[0])

            op_index = action.item()
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            embed = self.op_embedding(action)

            edit_seq.append((layer_index, op_index))

        return edit_seq, sum(logp_buf), sum(entropy_buf), max_entropy

class NASBench201ModificationController_DifferentPolicy(BaseController):
    def __init__(self, K=10, n_steps=2, gnn_layers=2, in_features=201,
                 n_layers=6, n_operations=5, hidden_size=64,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5,
                 batch_size=1, device=auto_device, no_edge=False):
        super(NASBench201ModificationController_DifferentPolicy, self).__init__(hidden_size, device)
        self.K = K
        self.n_steps = n_steps
        # self.has_resolution = has_resolution
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
        # if self.has_resolution:
        #     self.resolution_decision = nn.Linear(self.hidden_size, N_AVAILABLE_RESOLUTIONS)
        #     self.resolution_embedding = nn.Embedding(n_operations, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm0 = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def forward(self, graph: DynamicGraph, force_uniform=False):
        node_features = graph.node_features.to(device=self.device)
        edge_features = graph.edge_features.to(device=self.device)
        if self.no_edge:
            # print("no_edge")
            with torch.no_grad():
                edge_features = edge_features.detach()
                edge_features.requires_grad = False
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
            hx, cx = self.lstm0(embed, hidden)
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
            if force_uniform:
                logits = torch.zeros(self.n_operations, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.op_decision(hx).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            probs = F.softmax(logits, dim=-1)
            # logger.debug(f"Wchih Operation: {probs.tolist()}")
            action, select_log_p, entropy = self._impl(probs)
            max_entropy += math.log(probs.shape[0])

            op_index = action.item()
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            embed = self.op_embedding(action)

            edit_seq.append((layer_index, op_index))

        return edit_seq, sum(logp_buf), sum(entropy_buf), max_entropy

class NASBench201ModificationController_SelectBestSpace(BaseController):
    def __init__(self, K=10, n_steps=2, gnn_layers=2, in_features=201,
                 n_layers=6, n_operations=5, hidden_size=64,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5,
                 batch_size=1, device=auto_device, no_edge=False,
                 dataset=None):
        super(NASBench201ModificationController_SelectBestSpace, self).__init__(hidden_size, device)
        self.dataset = dataset
        self.K = K
        self.n_steps = n_steps
        # self.has_resolution = has_resolution
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
        # if self.has_resolution:
        #     self.resolution_decision = nn.Linear(self.hidden_size, N_AVAILABLE_RESOLUTIONS)
        #     self.resolution_embedding = nn.Embedding(n_operations, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def forward(self, graph: DynamicGraph, force_uniform=False):
        node_features = graph.node_features.to(device=self.device)
        edge_features = graph.edge_features.to(device=self.device)
        if self.no_edge:
            # print("no_edge")
            with torch.no_grad():
                edge_features = edge_features.detach()
                edge_features.requires_grad = False
                edge_features.zero_()
        graph_features, _ = self.gnn(node_features, edge_features, graph.edge_index)

        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)

        edit_seq = []
        logp_buf = []
        entropy_buf = []
        max_entropy = 0

        # decide which node is most potential
        # embed = graph_features.mean(dim=0, keepdim=True)
        # if force_uniform:
        #     logits = torch.zeros(self.K, device=self.device)
        # else:
        #     hx, cx = self.lstm(embed, hidden)
        #     hidden = (hx, cx)
        #     query = torch.tanh(self.emb_attn(graph_features) + self.hid_attn(hx))
        #     logits = self.v_attn(query).view(-1)  # (n_nodes,)
        #     logits = self._scale_attention(logits, self.temperature, self.tanh_constant)
        # probs = F.softmax(logits, dim=-1)
        # # logger.debug(f"Most Potential Node: {probs.tolist()}")
        # action, select_log_p, entropy = self._impl(probs)
        # max_entropy += math.log(probs.shape[0])

        best_index = 0
        best_acc = 0
        for i,node in enumerate(graph.nodes):
            if node.metadata[self.dataset]["val_acc"] > best_acc:
                best_acc = node.metadata[self.dataset]["val_acc"]
                best_index = i
        edit_seq.append(best_index)
        action = torch.tensor([best_index])
        # print(action.shape)
        # logp_buf.append(select_log_p)
        # entropy_buf.append(entropy)

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
            if force_uniform:
                logits = torch.zeros(self.n_operations, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.op_decision(hx).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            probs = F.softmax(logits, dim=-1)
            # logger.debug(f"Wchih Operation: {probs.tolist()}")
            action, select_log_p, entropy = self._impl(probs)
            max_entropy += math.log(probs.shape[0])

            op_index = action.item()
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            embed = self.op_embedding(action)

            edit_seq.append((layer_index, op_index))

        return edit_seq, sum(logp_buf), sum(entropy_buf), max_entropy