import json
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch
from typing import Dict


class Data(object):
    def __init__(self, adj, edge_list, features, labels):
        self.adj = adj
        self.edge_list = edge_list
        self.features = features
        self.labels = labels
        self.num_features = features.size(1)
        self.num_classes = labels.size(1)

    def to(self, device):
        self.adj = self.adj.to(device)
        self.edge_list = self.edge_list.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)


def load_data(dataset_str: str) -> Data:
    if dataset_str in ['ppi']:
        data = load_ppi_data()
    else:
        data = load_ppi_graphsage_data()
    return data


def load_ppi_subdata(dir: str, data_type: str) -> Data:
    G = nx.json_graph.node_link_graph(json.load(open(dir + data_type + "_graph.json")))
    features = torch.FloatTensor(np.load(dir + data_type + "_feats.npy"))
    labels = torch.FloatTensor(np.load(dir + data_type + "_labels.npy"))
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    edge_list = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)
    return Data(adj, edge_list, features, labels)


def load_ppi_data() -> Dict[str, Data]:
    dir = "data/ppi/"
    train_data = load_ppi_subdata(dir, "train")
    val_data = load_ppi_subdata(dir, "valid")
    test_data = load_ppi_subdata(dir, "test")
    return {"train": train_data,
            "val": val_data,
            "test": test_data}


def load_ppi_graphsage_data():
    dir = "data/ppi/"
    G = nx.json_graph.node_link_graph(json.load(open(dir + "ppi-G.json")))
    val_ids = [n for n in G.nodes() if G.nodes[n]['val']]
    test_ids = [n for n in G.nodes() if G.nodes[n]['test']]
    train_ids = [n for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']]

    labels = json.load(open(dir + "ppi-class_map.json"))
    train_labels = torch.LongTensor([labels[str(n)] for n in train_ids])
    val_labels = torch.LongTensor([labels[str(n)] for n in val_ids])
    test_labels = torch.LongTensor([labels[str(n)] for n in test_ids])

    feats = np.load(dir + "ppi-feats.npy")
    train_features = torch.FloatTensor([feats[n] for n in train_ids])
    val_features = torch.FloatTensor([feats[n] for n in val_ids])
    test_features = torch.FloatTensor([feats[n] for n in test_ids])


def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list):
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj


def preprocess_features(features):
    rowsum = features.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1
    features = features / rowsum
    return features
