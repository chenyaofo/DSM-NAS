import torch.nn as nn


class MultilayerPerceptron(nn.Sequential):
    def __init__(self, n_layers, in_features, hidden_features, out_features, drop_rate=0.5):
        assert n_layers > 1
        layers = []
        for i in range(n_layers):
            first, last = i == 0, i == n_layers-1
            in_f = in_features if first else hidden_features
            out_f = out_features if last else hidden_features
            layers.append(nn.Linear(in_f, out_f))
            if not last:
                if drop_rate is not None:
                    layers.append(nn.Dropout(drop_rate))
                layers.append(nn.ReLU(inplace=True))
        super(MultilayerPerceptron, self).__init__(*layers)
