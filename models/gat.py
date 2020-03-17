import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAT(nn.Module):
    def __init__(self, nfeat, nclass, nhid, nhead, nhead_out, alpha, dropout):
        super(GAT, self).__init__()
        self.atts1 = [GATConv(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nhead)]
        self.atts2 = [GATConv(nhid * nhead, nhid, dropout=dropout, alpha=alpha) for _ in range(nhead)]
        self.out_atts = [GATConv(nhid * nhead, nclass, dropout=dropout, alpha=alpha) for _ in range(nhead_out)]
        for i, att in enumerate(self.atts1):
            self.add_module('attention1_{}'.format(i), att)
        for i, att in enumerate(self.atts2):
            self.add_module('attention2_{}'.format(i), att)
        for i, att in enumerate(self.out_atts):
            self.add_module('out_att{}'.format(i), att)
        self.reset_parameters()

    def reset_parameters(self):
        for att in self.atts1:
            att.reset_parameters()
        for att in self.atts2:
            att.reset_parameters()
        for att in self.out_atts:
            att.reset_parameters()

    def forward(self, data):
        x, edge_list = data.features, data.edge_list
        x = torch.cat([att(x, edge_list) for att in self.atts1], dim=1)
        x = F.elu(x)
        x = torch.cat([att(x, edge_list) for att in self.atts2], dim=1)
        x = F.elu(x)
        x = torch.sum(torch.stack([att(x, edge_list) for att in self.out_atts]), dim=0) / len(self.out_atts)
        return torch.sigmoid(x)


def sp_softmax(indices, values, N):
    source, _ = indices
    v_max = values.max()
    exp_v = torch.exp(values - v_max)
    exp_sum = torch.zeros(N, 1, device=device)
    exp_sum.scatter_add_(0, source.unsqueeze(1), exp_v)
    softmax_v = exp_v / exp_sum[source]
    return softmax_v


def sp_matmul(indices, values, mat):
    source, target = indices
    out = torch.zeros_like(mat)
    out.scatter_add_(0, source.expand(mat.size(1), -1).t(), values * mat[target])
    return out


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, bias=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_list):
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.matmul(x, self.weight)

        source, target = edge_list
        a_input = torch.cat([h[source], h[target]], dim=1)
        e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=self.alpha)
        attention = sp_softmax(edge_list, e, h.size(0))

        attention = F.dropout(attention, self.dropout, training=self.training)
        h = F.dropout(h, self.dropout, training=self.training)
        h_prime = sp_matmul(edge_list, attention, h)
        if self.bias is not None:
            h_prime = h_prime + self.bias

        return h_prime


def create_gat_model(nfeat, nclass, nhid=256, nhead=4, nhead_out=6, alpha=0.2, dropout=0.6):
    model = GAT(nfeat, nclass, nhid, nhead, nhead_out, alpha=alpha, dropout=dropout)
    return model
