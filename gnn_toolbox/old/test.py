from torch_geometric.datasets import KarateClub, Planetoid, TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
import os
import scipy.sparse as sp
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
# from config_def import cfg
dataset = KarateClub()
mutag = TUDataset(root='./datasets', name='MUTAG')
other_data = Planetoid(root='./datasets', name='cora')
# benchmark = GNNBenchmarkDataset(root='./datasets', name='ogbn-arxiv')
nodeproppred = PygNodePropPredDataset(name='ogbn-arxiv', root='./datasets')
nodeproppred = nodeproppred[0]
# print('nodeproppred', nodeproppred)
data= other_data[0]
data2 = other_data[0]
n = data.num_nodes
data.edge_weight = torch.ones(data.edge_index.size(1))
adj_deep = sp.csr_matrix((data.edge_weight, data.edge_index), (data.num_nodes, data.num_nodes))

adj_rob = sp.csr_matrix((np.ones(data2.edge_index.shape[1]),
                                  (data2.edge_index[0], data2.edge_index[1])), shape=(n, n))
transform = T.Compose([T.ToUndirected(), T.ToSparseTensor(layout=torch.sparse_csr)])
data5= transform(nodeproppred)
print('data5', data5)
from torch_geometric.utils.sparse import is_sparse
print('is_sparse', is_sparse(data5.adj_t))
# print('adj_deep', adj_deep)
# print('adj_rob', adj_rob)
# print(dataset.num_classes)
# print(torch.unique(dataset._data.y).shape[0])
# # print(torch.unique(dataset._data.y).shape[1])
# print(other_data._data.keys())
# # for key in other_data._data.keys():
# #     print('key', key)
    
# for key in other_data.data.keys():
#     print('key', key)
# print(os.cpu_count())
# print( torch.cuda.device_count())

# split_keys = {"val", "test"}
# print(other_data._data)
# print(other_data.get_idx_split())
# print('_data', sum(key in other_data._data for key in split_keys))
# print('data', sum(key in other_data.data for key in split_keys))
# loader = DataLoader(other_data, batch_size=32, shuffle=True, mask='train_mask')
# print('1',mutag.edge_index)
# print('1',mutag[0].edge_index)
# print('2',other_data[0].train_mask.shape)
# print(other_data[0].items())
# print(other_data.train_mask.nonzero().squeeze())
# print(other_data.train_mask.nonzero().shape)
# print(issparse(other_data.edge_index))
# print(other_data._data.is_sparse())
# for data in loader:
#     assert torch.all(data.test_mask == other_data.data.test_mask) 

# cfg.freeze()
# cfg.trainer.what = 'hha'
# print('cfg', cfg)