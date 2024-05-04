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

    if hasattr(dataset, 'get_idx_split'):
        split = dataset.get_idx_split()
    else:
        split = dict(
            train=data.train_mask.nonzero().squeeze(),
            valid=data.val_mask.nonzero().squeeze(),
            test=data.test_mask.nonzero().squeeze()
        )

    # converting to numpy arrays, so we don't have to handle different
    # array types (tensor/numpy/list) later on.
    # Also we need numpy arrays because Numba cant determine type of torch.Tensor
    split = {k: v.numpy() for k, v in split.items()}

    edge_index = data.edge_index.cpu()
    if data.edge_attr is None:
        edge_weight = torch.ones(edge_index.size(1))
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
    print(adj)
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
from customize.architecture import GCN
dataset = Planetoid(name = 'cora', root = './datasets', transform=T.ToSparseTensor())
dataset1= Planetoid(name = 'cora', root = './datasets')
data = dataset1[0]
print('data.edge_index', data.edge_index)
data2 = dataset[0]
# print(a)
adj_t = data2.adj_t
# print('data2.edge_index', data2.adj_t)

row, col, edge_attr = adj_t.t().coo()
edge_index = torch.stack([row, col], dim=0)
row2, col2, edge_attr2 = adj_t.coo()
edge_index2 = torch.stack([row2, col2], dim=0)
print('data2 edge_index', edge_index)   
print('are they equal:', torch.equal(edge_index, edge_index2))
model1= GCN(in_channels=dataset.num_features, out_channels=dataset.num_classes, hidden_channels=16, num_layers=2, dropout=0.5)
model2 = GCN(in_channels=dataset.num_features, out_channels=dataset.num_classes, hidden_channels=16, num_layers=2, dropout=0.5)

def train(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()

@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index)
    return float(accuracy(pred, data.y, data.test_mask))

train(model1, data)
print('model1 acc', test(model1, data))
attr, adj, labels, splits, n = prep_graph(dataset1, make_undirected=False)

def train2(model, attr, adj, split, label, epochs=200, lr=0.01, weight_decay=5e-4):
    model = model.to('cuda')
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(attr, adj.t())
        loss = F.cross_entropy(pred[split['train']], label[split['train']])
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        model.eval()
        pred = model(attr, adj.t())
        acc = float(accuracy(pred, label, split['test']))
        print('model2 acc', acc)


print(train2(model2, attr, adj, splits, labels))

