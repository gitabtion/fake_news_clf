import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, layer_num: int, size=64):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([InitializedConv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([InitializedConv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=0.1, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            # x = F.relu(x)
        return x


class InitializedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                             bias=bias)
        self.relu = relu
        if self.relu:
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        if self.relu:
            return F.relu(self.out(x))
        else:
            return self.out(x)


class TorchFM(nn.Module):
    def __init__(self, input_dim):
        super(TorchFM, self).__init__()
        connector_dim = input_dim
        w0 = torch.zeros(1).double()
        w = torch.zeros(connector_dim).double()
        v = torch.empty(connector_dim, 10).double()
        nn.init.xavier_normal_(v)

        self.w0 = nn.Parameter(w0)
        self.w = nn.Parameter(w)
        self.v = nn.Parameter(v)

    def forward(self, x):
        x = x.double()
        linear_terms = torch.add(self.w0, torch.sum(torch.mul(self.w, x), axis=1))
        pair_interactions = 0.5 * torch.sum(torch.sub(torch.pow(torch.matmul(x, self.v), 2),
                                                      torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2))), axis=1)

        out = torch.add(linear_terms, pair_interactions)
        return out


class NeuralTensorNet(nn.Module):
    def __init__(self):
        super(NeuralTensorNet, self).__init__()
        T = torch.empty(768, 768)
        nn.init.xavier_normal_(T)
        W = torch.empty(768 * 2, 1)
        nn.init.xavier_normal_(W)
        self.b = nn.Parameter(torch.zeros(1))
        self.T = nn.Parameter(T)
        self.W = nn.Parameter(W)

    def forward(self, content, tag):
        x1 = torch.sum(torch.matmul(content, self.T) * tag, dim=1).unsqueeze(1)
        x2 = torch.matmul(torch.cat([content, tag], dim=1), self.W)
        out = F.relu(x1 + x2 + self.b).squeeze()
        return out


class NeuralTensorFM(nn.Module):
    def __init__(self):
        super(NeuralTensorFM, self).__init__()
        T = torch.empty(768, 768)
        nn.init.xavier_normal_(T)
        W = torch.empty(768 * 2, 1)
        nn.init.xavier_normal_(W)
        V = torch.empty(768 * 2, 10)
        nn.init.xavier_normal_(V)
        self.b = nn.Parameter(torch.zeros(1))
        self.T = nn.Parameter(T)
        self.W = nn.Parameter(W)
        self.V = nn.Parameter(V)
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, content, tag):
        x1 = torch.sum(torch.matmul(content, self.T) * tag, dim=1).unsqueeze(1)
        x = torch.cat([content, tag], dim=1)
        x2 = torch.matmul(x, self.W)
        pair_interactions = 0.5 * torch.sum(torch.sub(torch.pow(torch.matmul(x, self.V), 2),
                                                      torch.matmul(torch.pow(x, 2), torch.pow(self.V, 2))),
                                            axis=1).unsqueeze(1)
        y = torch.matmul(torch.cat([x1, x2, pair_interactions], dim=1), self.weights)
        output = F.relu(y + self.b).squeeze()
        return output
