# class BaseModel():
#     @ staticmethod
#     def parse_forward_input(data: Optional[Union[Data, TensorType["n_nodes", "n_features"]]] = None,
#                             adj: Optional[Union[SparseTensor,
#                                                 torch.sparse.FloatTensor,
#                                                 Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
#                                                 TensorType["n_nodes", "n_nodes"]]] = None,
#                             attr_idx: Optional[TensorType["n_nodes", "n_features"]] = None,
#                             edge_idx: Optional[TensorType[2, "nnz"]] = None,
#                             edge_weight: Optional[TensorType["nnz"]] = None,
#                             n: Optional[int] = None,
#                             d: Optional[int] = None) -> Tuple[TensorType["n_nodes", "n_features"],
#                                                               TensorType[2, "nnz"],
#                                                               TensorType["nnz"]]:
#         edge_weight = None
#         # PyTorch Geometric support
#         if isinstance(data, Data):
#             x, edge_idx = data.x, data.edge_index
#         # Randomized smoothing support
#         elif attr_idx is not None and edge_idx is not None and n is not None and d is not None:
#             x = coalesce(attr_idx, torch.ones_like(attr_idx[0], dtype=torch.float32), m=n, n=d)
#             x = torch.sparse.FloatTensor(x[0], x[1], torch.Size([n, d])).to_dense()
#             edge_idx = edge_idx
#         # Empirical robustness support
#         elif isinstance(adj, tuple):
#             # Necessary since `torch.sparse.FloatTensor` eliminates the gradient...
#             x, edge_idx, edge_weight = data, adj[0], adj[1]
#         elif isinstance(adj, SparseTensor):
#             x = data

#             edge_idx_rows, edge_idx_cols, edge_weight = adj.coo()
#             edge_idx = torch.stack([edge_idx_rows, edge_idx_cols], dim=0)
#         else:
#             if not adj.is_sparse:
#                 adj = adj.to_sparse()

#             x, edge_idx, edge_weight = data, adj._indices(), adj._values()

#         if edge_weight is None:
#             edge_weight = torch.ones_like(edge_idx[0], dtype=torch.float32)

#         if edge_weight.dtype != torch.float32:
#             edge_weight = edge_weight.float()

#         return x, edge_idx, edge_weight
    
    
    
    
#     model(x, edge_idx, edge_weight)
    
#     depending on the attack, 