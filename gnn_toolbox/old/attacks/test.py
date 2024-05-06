from torch_geometric.datasets import KarateClub, Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import (
    is_undirected,
    sort_edge_index,
    to_torch_csr_tensor,
    to_undirected,
)
from sacred import Experiment
# transform = T.Compose([T.to_undirected()])
dataset = KarateClub()
data = dataset[0]
data.edge_index = to_undirected(data.edge_index)
print(data.keys())

ex = Experiment()

@ex.config
def config():
    overwrite = None

    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    # default params
    data_dir = './data'
    dataset = 'cora_ml'
    make_undirected = True
    binary_attr = False
    data_device = 0

    device = 0
    seed = 0

    attack = 'DICE'
    attack_params = dict(
        epochs=500,
        fine_tune_epochs=100,
        keep_heuristic="WeightOnly",
        block_size=100_000,
        do_synchronize=True,
        loss_type="tanhMargin",
    )
    epsilons = [0.01, 0.1]

    artifact_dir = 'cache'
    model_label = "GCN"
    model_storage_type = 'pretrained'
    pert_adj_storage_type = 'evasion_global_adj'
    pert_attr_storage_type = 'evasion_global_attr'

    debug_level = "info"