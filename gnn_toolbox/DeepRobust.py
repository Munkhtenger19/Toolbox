import numpy as np
import torch
import scipy.sparse as sp
from itertools import repeat
import os.path as osp
import warnings
import sys
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid
from torch_geometric.seed import seed_everything
class Dpr2Pyg(InMemoryDataset):
    """Convert deeprobust data (sparse matrix) to pytorch geometric data (tensor, edge_index)

    Parameters
    ----------
    dpr_data :
        data instance of class from deeprobust.graph.data, e.g., deeprobust.graph.data.Dataset,
        deeprobust.graph.data.PtbDataset, deeprobust.graph.data.PrePtbDataset
    transform :
        A function/transform that takes in an object and returns a transformed version.
        The data object will be transformed before every access. For example, you can
        use torch_geometric.transforms.NormalizeFeatures()

    Examples
    --------
    We can first create an instance of the Dataset class and convert it to
    pytorch geometric data format.

    >>> from deeprobust.graph.data import Dataset, Dpr2Pyg
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> pyg_data = Dpr2Pyg(data)
    >>> print(pyg_data)
    >>> print(pyg_data[0])
    """

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/'  # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        dpr_data = self.dpr_data
        edge_index = torch.LongTensor(dpr_data.adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(dpr_data.features):
            x = torch.FloatTensor(dpr_data.features.todense()).float()
        else:
            x = torch.FloatTensor(dpr_data.features).float()
        y = torch.LongTensor(dpr_data.labels)
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        data = Data(x=x, edge_index=edge_index, y=y)
        train_mask = index_to_mask(idx_train, size=y.size(0))
        val_mask = index_to_mask(idx_val, size=y.size(0))
        test_mask = index_to_mask(idx_test, size=y.size(0))
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def update_edge_index(self, adj):
        """ This is an inplace operation to substitute the original edge_index
        with adj.nonzero()

        Parameters
        ----------
        adj: sp.csr_matrix
            update the original adjacency into adj (by change edge_index)
        """
        self.data.edge_index = torch.LongTensor(adj.nonzero())
        self.data, self.slices = self.collate([self.data])

    def get(self, idx):
        if self.slices is None:
            return self.data
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                        slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass


class Pyg2Dpr(InMemoryDataset):
    """Convert pytorch geometric data (tensor, edge_index) to deeprobust
    data (sparse matrix)

    Parameters
    ----------
    pyg_data :
        data instance of class from pytorch geometric dataset

    Examples
    --------
    We can first create an instance of the Dataset class and convert it to
    pytorch geometric data format and then convert it back to Dataset class.

    >>> from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> pyg_data = Dpr2Pyg(data)
    >>> print(pyg_data)
    >>> print(pyg_data[0])
    >>> dpr_data = Pyg2Dpr(pyg_data)
    >>> print(dpr_data.adj)
    """

    def __init__(self, pyg_data, **kwargs):
        is_ogb = hasattr(pyg_data, 'get_idx_split')
        if is_ogb:  # get splits for ogb datasets
            splits = pyg_data.get_idx_split()
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes
        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
                                  (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()
        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1)  # ogb-arxiv needs to reshape
        if is_ogb:  # set splits for ogb datasets
            self.idx_train = splits['train'].numpy()
            self.idx_val = splits['valid'].numpy()
            self.idx_test = splits['test'].numpy()
        else:
            try:
                self.idx_train = mask_to_index(pyg_data.train_mask, n)
                self.idx_val = mask_to_index(pyg_data.val_mask, n)
                self.idx_test = mask_to_index(pyg_data.test_mask, n)
            except AttributeError:
                print(
                    'Warning: This pyg dataset is not associated with any data splits...')
        self.name = 'Pyg2Dpr'

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

import torch
import numpy as np
import torch_sparse
import time
import torch.nn.functional as F


def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        print('ahaha_adj')
        adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to(device)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        print('ahaha_feature')
        features = torch.from_numpy(features).to(device)
    else:
        features = torch.from_numpy(features).to(device)
        # features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)

seed_everything(0)
dataset = Planetoid(name = 'cora', root = './datasets')
dpr_data = Pyg2Dpr(dataset)
# print(dataset[0].items())
from DeepRobust.DICE import DICE
from torch_geometric.nn import GCN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
# print(GCN.num_classes)
model = DICE(model=gcn, device=device)
n_perturbations = int(0.05 * (dpr_data.adj.sum()//2))

model.attack(dpr_data.adj, dpr_data.labels, n_perturbations)
modified_adj = model.modified_adj
dpr_data.adj = modified_adj
adj, features, labels = to_tensor(dpr_data.adj, dpr_data.features, dpr_data.labels)
train_mask = index_to_mask(dpr_data.idx_train, labels.size(0))
val_mask = index_to_mask(dpr_data.idx_val, labels.size(0))
test_mask = index_to_mask(dpr_data.idx_test, labels.size(0))

dpr_data = Dpr2Pyg(dpr_data)

# Now I can train the model with the modified data since I converted back to torch sparse and GCN accept torch sparse


optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01) 
# DataLoader(d, batch_size=32, shuffle=True, mask='train_mask')

def train(x, edge_index, y, train_mask):
    gcn.train()
    optimizer.zero_grad()
    out = gcn(x, edge_index)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(x, edge_index, y, train_mask, val_mask, test_mask):
    model.eval()
    pred = gcn(x, edge_index).argmax(dim=-1)

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        accs.append(int((pred[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

def run_experiment(features, adj, labels, train_mask, val_mask, test_mask):
    best_val_acc = test_acc = 0
    times = []
    
    for epoch in range(1, 200 + 1):
        start = time.time()
        loss = train(features, adj, labels, train_mask)
        train_acc, val_acc, tmp_test_acc = test(features, adj, labels, train_mask, val_mask, test_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        # log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
            f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        times.append(time.time() - start)
    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
    
run_experiment(features, adj, labels, train_mask, val_mask, test_mask)

for module in gcn.modules():
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()

run_experiment(dpr_data[0].x, dpr_data[0].edge_index, dpr_data[0].y, dpr_data[0].train_mask, dpr_data[0].val_mask, dpr_data[0].test_mask)                   