import torch
import logging
from typing import Iterable, Union, List
from torch_geometric.datasets import Planetoid
# class PyG2SciSparse:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.data = self.dataset[0]
        
#     def to_sci_sparse(self, make_undirected=True, return_original_split=False):
#         if hasattr(self.data, '__num_nodes__'):
#             self.num_nodes = self.data.__num_nodes__
#         else:
#             self.num_nodes = self.data.num_nodes

#         if hasattr(self.dataset, 'get_idx_split'):
#             self.split = dataset.get_idx_split()
#         else:
#             try:
#                 self.split = dict(
#                     train=self.data.train_mask.nonzero().squeeze(),
#                     valid=self.data.val_mask.nonzero().squeeze(),
#                     test=self.data.test_mask.nonzero().squeeze()
#                 )
#             except AttributeError:
#                 print(
#                     'Warning: This pyg dataset is not associated with any data splits...')
#                 # ! Make custom split available
#                 self.split = dict(
#                     train =  self.mask_to_index(data.train_mask, self.num_nodes),
#                     valid =  self.mask_to_index(data.val_mask, self.num_nodes),
#                     test =  self.mask_to_index(data.test_mask, self.num_nodes),
#                 )
#         self.split = {k: v.numpy() for k, v in self.split.items()}

#         self.edge_index = self.data.edge_index.cpu()
#         if self.data.edge_attr is None:
#             self.edge_weight = torch.ones(self.edge_index.size(1))
#         else:
#             self.edge_weight = self.data.edge_attr
#         self.edge_weight = self.edge_weight.cpu()

#         self.adj = sp.csr_matrix((self.edge_weight, self.edge_index), (self.num_nodes, self.num_nodes))

#         del edge_index
#         del edge_weight

#         # make unweighted
#         adj.data = np.ones_like(adj.data)

#         if make_undirected:
#             adj = utils.to_symmetric_scipy(adj)

#             logging.debug("Memory Usage after making the graph undirected:")
#             logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

#         logging.debug("Memory Usage after normalizing the graph")
#         logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

#         # adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to(device)

#         attr_matrix = data.x.cpu().numpy()

#         attr = torch.from_numpy(attr_matrix).to(device)

#         logging.debug("Memory Usage after normalizing graph attributes:")
#         logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

#         labels = data.y.squeeze().to(device)
#         if return_original_split and split is not None:
#             return attr, adj, labels, split

#         return attr, adj, labels, None
    
#     def mask_to_index(self, index, size):
#         all_idx = np.arange(size)
#         return all_idx[index]
    
#     # @typechecked
#     def train_val_test_split_tabular(self,
#             *arrays: Iterable[Union[np.ndarray, sp.spmatrix]],
#             train_size: float = 0.5,
#             val_size: float = 0.3,
#             test_size: float = 0.2,
#             stratify: np.ndarray = None,
#             random_state: int = None
#     ) -> List[Union[np.ndarray, sp.spmatrix]]:
#         """Split the arrays or matrices into random train, validation and test subsets.

#         Parameters
#         ----------
#         *arrays
#                 Allowed inputs are lists, numpy arrays or scipy-sparse matrices with the same length / shape[0].
#         train_size
#             Proportion of the dataset included in the train split.
#         val_size
#             Proportion of the dataset included in the validation split.
#         test_size
#             Proportion of the dataset included in the test split.
#         stratify
#             If not None, data is split in a stratified fashion, using this as the class labels.
#         random_state
#             Random_state is the seed used by the random number generator;

#         Returns
#         -------
#         list, length=3 * len(arrays)
#             List containing train-validation-test split of inputs.

#         """
#         if len(set(array.shape[0] for array in arrays)) != 1:
#             raise ValueError("Arrays must have equal first dimension.")
#         idx = np.arange(arrays[0].shape[0])
#         idx_train_and_val, idx_test = train_test_split(idx,
#                                                     random_state=random_state,
#                                                     train_size=(train_size + val_size),
#                                                     test_size=test_size,
#                                                     stratify=stratify)
#         if stratify is not None:
#             stratify = stratify[idx_train_and_val]
#         idx_train, idx_val = train_test_split(idx_train_and_val,
#                                             random_state=random_state,
#                                             train_size=(train_size / (train_size + val_size)),
#                                             test_size=(val_size / (train_size + val_size)),
#                                             stratify=stratify)
#         result = []
#         for X in arrays:
#             result.append(X[idx_train])
#             result.append(X[idx_val])
#             result.append(X[idx_test])
#         return result
    
    
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Union
from torchtyping import TensorType
import torch_sparse
from torch_sparse import SparseTensor
# from gnn_toolbox.custom_modules.attacks.dice import DICE
    
def prep_graph(dataset,
               device: Union[int, str, torch.device] = 0,
               make_undirected: bool = True,
            #    binary_attr: bool = False,
            #    feat_norm: bool = False,
               ) -> Tuple[TensorType["num_nodes", "num_features"],
                        SparseTensor,
                        TensorType["num_nodes"],
                        Optional[Dict[str, np.ndarray]]]:
    """Prepares and normalizes the desired dataset

    Parameters
    ----------
    name : str
        Name of the data set. One of: `cora_ml`, `citeseer`, `pubmed`
    device : Union[int, torch.device]
        `cpu` or GPU id, by default 0
    binary_attr : bool, optional
        If true the attributes are binarized (!=0), by default False
    dataset_root : str, optional
        Path where to find/store the dataset, by default "datasets"
    return_original_split: bool, optional
        If true (and the split is available for the choice of dataset) additionally the original split is returned.

    Returns
    -------
    Tuple[torch.Tensor, torch_sparse.SparseTensor, torch.Tensor]
        dense attribute tensor, sparse adjacency matrix (normalized) and labels tensor.
    """
    split = None

    logging.debug("Memory Usage before loading the dataset:")
    # logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    data = dataset[0]

    if hasattr(data, '__num_nodes__'):
        num_nodes = data.__num_nodes__
    else:
        num_nodes = data.num_nodes
    
    # if hasattr(dataset, 'get_idx_split'):
    #     split = dataset.get_idx_split()
    # else:
    #     split = dict(
    #         train=data.train_mask.nonzero().squeeze(),
    #         valid=data.val_mask.nonzero().squeeze(),
    #         test=data.test_mask.nonzero().squeeze()
    #     )

    # converting to numpy arrays, so we don't have to handle different
    # array types (tensor/numpy/list) later on.
    # Also we need numpy arrays because Numba cant determine type of torch.Tensor
    # split = {k: v.numpy() for k, v in split.items()}

    edge_index = data.edge_index.cpu()
    print('edge_index', edge_index)
    if data.edge_attr is None:
        # edge_weight = torch.ones(edge_index.size(1))
        edge_weight = torch.full((edge_index.size(1),), 2.0)
    else:
        edge_weight = data.edge_attr
    edge_weight = edge_weight.cpu()

    adj = sp.csr_matrix((edge_weight, edge_index), (num_nodes, num_nodes))

    del edge_index
    del edge_weight

    # make unweighted
    adj.data = np.ones_like(adj.data)

    if make_undirected:
        adj = to_symmetric_scipy(adj)
        logging.debug("Memory Usage after making the graph undirected:")
        # logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))
        num_edges = adj.nnz / 2
    else:
        num_edges = adj.nnz  

    logging.debug("Memory Usage after normalizing the graph")
    # logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to(device)

    attr_matrix = data.x.cpu().numpy()
    print('prep_graph adj', adj)
    attr = torch.from_numpy(attr_matrix).to(device)

    logging.debug("Memory Usage after normalizing graph attributes:")
    # logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    labels = data.y.squeeze().to(device)
    # labels = data.y.numpy()
    if split is not None:
        return attr, adj, labels, split, num_edges

    return attr, adj, labels, None, num_edges


def to_symmetric_scipy(adjacency: sp.csr_matrix):
    sym_adjacency = (adjacency + adjacency.T).astype(bool).astype(float)

    sym_adjacency.tocsr().sort_indices()

    return sym_adjacency

import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCN
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import QM9, CoraFull, Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import inspect
from gnn_toolbox.common import is_directed
coraa = CoraFull(root='./datasets', transform=T.ToSparseTensor())
cora = Planetoid(name='cora', root='./datasets')


# d = coraa[0]
# print('coraa', d)
# print('is_directed', is_directed(d.adj_t))
# init = inspect.signature(Planetoid )
# if 'name' in init.parameters.keys():
#     print('yesbitch')
    
# print(init.parameters.keys())
    
# for name, param in init.parameters.items():
#         print(f"- {name}: {param}")

# dataset2 = Planetoid(name = 'cora', root = './datasets')
# dataset5 = QM9(root='./datasets', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
# dataset6 = PygNodePropPredDataset(name='ogbn-arxiv', root='./datasets')
# data6= dataset6[0]
# print('qm9', data6)
# print('qm9.edge_attr', data6.edge_attr)
# from ogb.nodeproppred import PygNodePropPredDataset
# dataset3 = PygNodePropPredDataset(name='ogbn-arxiv', root='./datasets')
# dataset4 = PygNodePropPredDataset(name='ogbn-arxiv', root='./datasets', transform=T.ToUndirected())
# data = dataset[0]
# # print('pyg', data)
# data3 = dataset3[0]
# print('is_directed ogb', data3.is_directed())

# attr, adj, labels, splits, n = prep_graph(dataset6, make_undirected=False)
# row, col, edge_attr2 = adj.coo()
# edge_index = torch.stack([row, col], dim=0)
# print('after edge_attr', edge_attr2)
# print('pyg prep_graph adj', adj)
# attr, adj, labels, splits, n = prep_graph(dataset3, make_undirected=True)
# print('robu prep_graph adj', adj)

##############################

# data2 = dataset[0]
# print(data2.is_directed())
# adj_t = data2.adj_t
# # data.edge_index
# # model = DICE()
# amazon = Amazon(root='./datasets', name='computers')
# data9 = amazon[0]
def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()
# print(data9.is_directed())
# dataset2= PygNodePropPredDataset(name='ogbn-arxiv', root='./datasets', transform=T.ToSparseTensor(remove_edge_index=False))
dataset1= Planetoid(name = 'pubmed', root = './datasets')
data = dataset1[0]
train=data.train_mask.nonzero().squeeze()
model1= GCN(in_channels=dataset1.num_features, out_channels=dataset1.num_classes, hidden_channels=32, num_layers=2)
logits = model1(data.x, data.edge_index)
# print(accuracy(logits, data.y, train))
# print(accuracy(logits, data.y, data.train_mask))
from torch_geometric.datasets import PPI, KarateClub
dataset9 = QM9(root = './datasets', transform=T.ToSparseTensor(remove_edge_index=False))
data12 = dataset9[0]
print(data12)
print(data)

# print('pygnodepred', dataset1)
# we = dataset1[0]
# print('we', we)

# dataset2= Planetoid(name = 'citeseer', root = './datasets')
# dataset2= QM9(root = './datasets')
# data2 = dataset2[0]
# # train_mask=data2.train_mask.nonzero().squeeze()
# # attr, adj, labels, splits, n = prep_graph(dataset2, make_undirected=False)
# # print()
# from torch_geometric.utils import to_undirected, is_undirected
# print('is_undirected', is_undirected(data2.edge_index, num_nodes=data2.num_nodes))
# adj = SparseTensor(row=data2.edge_index[0], col=data2.edge_index[1], value=data2.edge_attr).to('cuda')
# adj2 = SparseTensor(row=data2.edge_index[0], col=data2.edge_index[1], value=data2.edge_weight if data2.edge_weight is not None else torch.ones(data2.edge_index.size(1))).to('cuda')
# print('num_edge',adj.nnz())
# print('what', adj)
# print('what2', adj.t())

# row, col, edge_attr = adj.t().coo()
# edge_index = torch.stack([row, col], dim=0)
# print('edge_index3', edge_index)
# print('edge_attr', edge_attr)
# # row, col, edge_attr = adj.coo()
# # edge_index = torch.stack([row, col], dim=0)
# # print('edge_index1', edge_index)


# edge_index = to_undirected(data2.edge_index)
# print('x',edge_index.shape[1])
# print('undirected', edge_index)
# edge_index2 = to_undirected(edge_index)
# print('undirected', edge_index)
# print('q',torch.equal(edge_index, edge_index2))
# print(torch.equal(data2.edge_index, edge_index))
# print("numm", data2.num_nodes)

# print("numm2", data2.x.shape[0])
# print("numm2", data2.x.size(0))

# print(T.ToSparseTensor.__name__)

# row, col, edge_attr = adj.coo()
# cora_edge_index = torch.stack([row, col], dim=0)
# print('edge_attr', edge_attr)
# data = dataset1[0]
# row, col, edge_attr = data.adj_t.coo()
# cora_edge_index = torch.stack([row, col], dim=0)

# # cora_edge_index = cora_edge_index.to('cpu')
# if edge_attr is None:
#     edge_attr = torch.ones(cora_edge_index.size(1))
# row2, col2, edge_attr2 = adj_t.coo()
# edge_index2 = torch.stack([row2, col2], dim=0)
# print('data2 edge_index', edge_index2)   
# print('are they equal:', torch.equal(edge_index, edge_index2))

import torch.nn as nn
from torch_geometric.nn import GCNConv

def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()

# @register_model("GCN")
class GCNWH(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int = 2, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.GCNConv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        self.layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.GCNConv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels)
        
    def forward(self, x, edge_index, edge_weight, **kwargs):
        x = self.GCNConv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.GCNConv2(x, edge_index, edge_weight)
        return x

def train3(model, attr, edge_index, data, label,edge_weight=None, epochs=200, lr=0.01, weight_decay=5e-4):
    model = model.to('cuda')
    res = []
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for module in model.children():
        module.reset_parameters()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(attr, edge_index, edge_weight)
        loss = F.cross_entropy(pred[data.train_mask], label[data.train_mask])
        acc = float(accuracy(pred, label, data.train_mask))
        res.append(acc)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        model.eval()
        pred = model(attr, edge_index, edge_weight)
        acc = float(accuracy(pred, label, data.train_mask))
        print('model2 acc', acc)
    return res

# print(train3(model1, data.x, cora_edge_index, edge_attr, data, data.y))
# from pprint import pprint
# res =train3(model1,attr, cora_edge_index, data2, labels, edge_attr)
# pprint(res)

# train2(model1,)
# model2 = GCN(in_channels=dataset.num_features, out_channels=dataset.num_classes, hidden_channels=16, num_layers=2, dropout=0.5)

def train(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()



@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index)
    return float(accuracy(pred, data.y, data.test_mask))

# train(model1, data)
# print('model1 acc', test(model1, data))
# attr, adj, labels, splits, n = prep_graph(dataset1, make_undirected=False)

def train2(model, attr, adj, split, label, epochs=200, lr=0.01, weight_decay=5e-4):
    model = model.to('cuda')
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(attr, adj)
        loss = F.cross_entropy(pred[split['train']], label[split['train']])
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        model.eval()
        pred = model(attr, adj.t())
        acc = float(accuracy(pred, label, split['test']))
        print('model2 acc', acc)


# print(train2(model2, attr, adj, splits, labels))

