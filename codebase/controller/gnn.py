import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _create_Pindex(n_nodes, edge_index):
    '''
    -1 represents the edge does not exist.
    '''

    rev = []
    _map = {v: i for i, v in enumerate(edge_index)}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            start = _map.get((i, j), -1)
            end = _map.get((j, i), -1)
            if start == -1 and end == -1:
                rev.append((-1, -1, -1))
            else:
                rev.append((i, start, end))
    return rev


class EdgeGraphSAGElayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,):
        '''
        implementation for 'Handling Missing Data with Graph Representation Learning'
        we make the model possible to handle with directed graph
        '''
        super(EdgeGraphSAGElayer, self).__init__()
        self.P_weights = Parameter(torch.Tensor(in_features*3, hidden_features))
        self.Q_weights = Parameter(torch.Tensor(in_features+hidden_features, out_features))
        self.W_weight = Parameter(torch.Tensor(in_features+hidden_features*2, out_features))
        self.pool_linear = nn.Linear(hidden_features, hidden_features)

    def forward(self, node_features, edge_features, edge_index):
        '''
        node_features: [n_nodes, n_features]
        edge_features: [n_edges, n_features]
        edge_index: [n_edges, 2]
        '''
        n_nodes, _ = node_features.shape
        n_edges, _ = edge_features.shape

        expand_node_features = self.expand_zeros(node_features)
        expand_edge_features = self.expand_zeros(edge_features)

        P_index = torch.tensor(_create_Pindex(n_nodes, edge_index), dtype=torch.long, device=node_features.device) + 1  # [n_nodes*(n_nodes-1), 3]
        P_node_index, P_edge_start_index, P_edge_end_index = torch.unbind(P_index, dim=1)  # [n_nodes*(n_nodes-1)+1,]
        Pselect_node_features = torch.index_select(expand_node_features, dim=0, index=P_node_index)  # [n_nodes*(n_nodes-1)+1, n_features]
        Pselect_edge_start_features = torch.index_select(expand_edge_features, dim=0, index=P_edge_start_index)  # [n_nodes*(n_nodes-1)+1, n_features]
        Pselect_edge_end_features = torch.index_select(expand_edge_features, dim=0, index=P_edge_end_index)  # [n_nodes*(n_nodes-1)+1, n_features]

        Pselect_features = torch.cat([Pselect_node_features, Pselect_edge_start_features, Pselect_edge_end_features], dim=1)
        Pfeatures = F.relu(torch.mm(Pselect_features, self.P_weights))
        # Pfeatures = Pfeatures.view((n_nodes, n_nodes-1, -1)).mean(dim=1, keepdim=False)  # [n_nodes, n_features]
        Pfeatures = F.relu(self.pool_linear(Pfeatures)).view((n_nodes, n_nodes-1, -1)).mean(dim=1, keepdim=False)

        Qfeatures = F.relu(torch.mm(torch.cat([node_features, Pfeatures], dim=1), self.Q_weights))  # [n_nodes, n_features]

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=node_features.device)
        start, end = torch.unbind(edge_index, dim=1)
        node_start_features = torch.index_select(Qfeatures, dim=0, index=start)
        node_end_features = torch.index_select(Qfeatures, dim=0, index=end)

        Wfeatures = F.relu(torch.mm(torch.cat([edge_features, node_start_features, node_end_features], dim=1), self.W_weight))  # [n_nodes, n_features]

        return Qfeatures, Wfeatures

    def expand_zeros(self, features):
        _, n_features = features.shape
        zeros = torch.zeros((1, n_features), dtype=features.dtype, device=features.device)
        return torch.cat((zeros, features), dim=0)


class EdgeGraphSAGE(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features):
        super(EdgeGraphSAGE, self).__init__()
        self.node_linear_transform = nn.Linear(in_features, hidden_features)
        self.edge_linear_transform = nn.Linear(in_features, hidden_features)

        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            _out_features = out_features if i == n_layers - 1 else hidden_features
            self.gnn_layers.append(EdgeGraphSAGElayer(hidden_features, hidden_features, _out_features))

    def forward(self, node_features, edge_features, edge_index):
        node_features = self.node_linear_transform(node_features)
        if edge_features is None:
            return node_features, None
        else:
            edge_features = self.edge_linear_transform(edge_features)

            for layer in self.gnn_layers:
                node_features, edge_features = layer(node_features, edge_features, edge_index)
            return node_features, edge_features


class NoEdgeGraph(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features):
        super(NoEdgeGraph, self).__init__()
        self.node_linear_transform = nn.Linear(in_features, out_features)
        # self.edge_linear_transform = nn.Linear(in_features, hidden_features)

        # self.gnn_layers = nn.ModuleList()
        # for i in range(n_layers):
        #     _out_features = out_features if i == n_layers - 1 else hidden_features
        #     self.gnn_layers.append(EdgeGraphSAGElayer(hidden_features, hidden_features, _out_features))

    def forward(self, node_features, edge_features, edge_index):
        node_features = self.node_linear_transform(node_features)
        return node_features, None
        # if edge_features is None:
        #     return node_features, None
        # else:
        #     edge_features = self.edge_linear_transform(edge_features)

        #     for layer in self.gnn_layers:
        #         node_features, edge_features = layer(node_features, edge_features, edge_index)
        #     return node_features, edge_features
