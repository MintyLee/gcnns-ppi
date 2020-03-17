import json
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch
from torch.utils.data import DataLoader
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
        exit()
    return data


def load_ppi_subdata(dir: str, data_type: str) -> DataLoader:
    G = nx.json_graph.node_link_graph(json.load(open(dir + data_type + "_graph.json")))
    graph_id = np.load(dir + data_type + "_graph_id.npy")
    features = torch.FloatTensor(np.load(dir + data_type + "_feats.npy"))
    labels = torch.FloatTensor(np.load(dir + data_type + "_labels.npy"))

    data_list = []
    id_set = list(np.unique(graph_id))
    for id in id_set:
        nodes = np.where(graph_id == id)[0]
        subG = G.subgraph(nodes)
        sub_feat = features[nodes]
        sub_labels = labels[nodes]
        coo_adj = nx.to_scipy_sparse_matrix(subG).tocoo()
        sub_edges = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
        sub_edges = sub_edges - sub_edges.min()
        sub_edges = add_self_loops(sub_edges, len(nodes))
        adj = normalize_adj(sub_edges)
        data_list.append(Data(adj, sub_edges, sub_feat, sub_labels))
    return DataLoader(data_list, batch_size=2, collate_fn=my_collate_fn)


def load_ppi_data() -> Dict[str, DataLoader]:
    dir = "data/ppi/"
    train_data = load_ppi_subdata(dir, "train")
    val_data = load_ppi_subdata(dir, "valid")
    test_data = load_ppi_subdata(dir, "test")
    return {"train": train_data,
            "val": val_data,
            "test": test_data}


def my_collate_fn(data_list):
    adj_indices = data_list[0].adj._indices()
    adj_values = data_list[0].adj._values()
    edge_list = data_list[0].edge_list
    for d in data_list[1:]:
        idx = edge_list.max() + 1
        adj_i = d.adj._indices() + idx
        adj_v = d.adj._values()
        adj_indices = torch.cat([adj_indices, adj_i], dim=1)
        adj_values = torch.cat([adj_values, adj_v], dim=0)
        edge_list = torch.cat([edge_list, d.edge_list + idx], dim=1)
    adj = torch.sparse.FloatTensor(adj_indices, adj_values)
    features = torch.cat([d.features for d in data_list], dim=0)
    labels = torch.cat([d.labels for d in data_list], dim=0)
    return Data(adj, edge_list, features, labels)


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
