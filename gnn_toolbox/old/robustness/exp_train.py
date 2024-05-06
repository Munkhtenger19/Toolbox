import numpy as np
from sacred import Experiment
import logging

# import numpy as np
# import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch
# from rgnn_at_scale.data import prep_graph, split
from attack.io import Storage
# from rgnn_at_scale.models import create_model, PPRGoWrapperBase  <--- CAN BE MADE BY MY FUNCTIONS
# from rgnn_at_scale.train import train
# from rgnn_at_scale.helper.utils import accuracy
# from rgnn_at_scale.helper import utils
import custom_modules.data

from gnn_toolbox.old.config_def import cfg
from gnn_toolbox.old.utils3 import create_model
import gnn_toolbox.old.utils3 as utils3
from typing import Any, Dict, Union, Tuple, Optional
from torchtyping import TensorType
from torch_sparse import SparseTensor
import torch_sparse
from custom_modules.data import load_dataset_from_cfg, set_dataset_info
import scipy.sparse as sp


ex = Experiment()

@ex.config
def config():
    # overwrite = None

    # if seml is not None:
    #     db_collection = None
    #     if db_collection is not None:
    #         ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    # default params
    data_dir = './datasets'
    dataset = 'cora'
    make_undirected = True
    binary_attr = False
    data_device = 0

    device = 0
    seed = 0

    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained'
    model_params = dict(
        label="Vanilla GCN",
        model="GCN",
        # hidden_channels=64,
    )

    train_params = dict(
        lr=1e-2,
        weight_decay=1e-3,
        patience=300,
        max_epochs=3000
    )

    ppr_cache_params = dict(
        data_artifact_dir="cache",
        data_storage_type="ppr"
    )

    display_steps = 100
    debug_level = "info"

def prep_graph(name: str,
               device: Union[int, str, torch.device] = 0,
               make_undirected: bool = True,
               binary_attr: bool = False,
               feat_norm: bool = False,
               dataset_root: str = 'data',
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
    logging.debug(utils3.get_max_memory_bytes() / (1024 ** 3))

    # if name in ['cora_ml', 'citeseer', 'pubmed']:
    #     attr, adj, labels = prep_cora_citeseer_pubmed(name, dataset_root, device, make_undirected)
    # elif name.startswith('ogbn'):
    # try:
    pyg_dataset = load_dataset_from_cfg()
    set_dataset_info(pyg_dataset)
    data = pyg_dataset[0]

    if hasattr(data, '__num_nodes__'):
        num_nodes = data.__num_nodes__
    else:
        num_nodes = data.num_nodes

    if hasattr(pyg_dataset, 'get_idx_split'):
        split = pyg_dataset.get_idx_split()
    else:
        print('puuu')
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
        adj = utils3.to_symmetric_scipy(adj)

        logging.debug("Memory Usage after making the graph undirected:")
        logging.debug(utils3.get_max_memory_bytes() / (1024 ** 3))

    logging.debug("Memory Usage after normalizing the graph")
    logging.debug(utils3.get_max_memory_bytes() / (1024 ** 3))

    adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to(device)

    attr_matrix = data.x.cpu().numpy()

    attr = torch.from_numpy(attr_matrix).to(device)

    logging.debug("Memory Usage after normalizing graph attributes:")
    logging.debug(utils3.get_max_memory_bytes() / (1024 ** 3))

    labels = data.y.squeeze().to(device)
    # except:
    #     raise NotImplementedError(f"Dataset `with name '{name}' is not supported")

    if binary_attr:
        # NOTE: do not use this for really large datasets.
        # The mask is a **dense** matrix of the same size as the attribute matrix
        attr[attr != 0] = 1
    elif feat_norm:
        attr = utils3.row_norm(attr)

    if split is not None:
        return attr, adj, labels, split

    return attr, adj, labels, None



def accuracy(logits: torch.Tensor, labels: torch.Tensor, split_idx: np.ndarray) -> float:
    """Returns the accuracy for a tensor of logits, a list of lables and and a split indices.

    Parameters
    ----------
    prediction : torch.Tensor
        [n x c] tensor of logits (`.argmax(1)` should return most probable class).
    labels : torch.Tensor
        [n x 1] target label.
    split_idx : np.ndarray
        [?] array with indices for current split.

    Returns
    -------
    float
        the Accuracy
    """
    return (logits.argmax(1)[split_idx] == labels[split_idx]).float().mean().item()

def train(model, attr, adj, labels, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, display_step=50):
    """Train a model using either standard training.
    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    adj: torch.Tensor [n, n]
        Dense adjacency matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: array-like [?]
        Indices of the training nodes.
    idx_val: array-like [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    Returns
    -------
    train_val, trace_val: list
        A tupole of lists of values of the validation loss during training.
    """
    trace_loss_train = []
    trace_loss_val = []
    trace_acc_train = []
    trace_acc_val = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf

    model.train()
    for it in tqdm(range(max_epochs), desc='Training...'):
        optimizer.zero_grad()

        logits = model(attr, adj)
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss_val = F.cross_entropy(logits[idx_val], labels[idx_val])

        loss_train.backward()
        optimizer.step()

        trace_loss_train.append(loss_train.detach().item())
        trace_loss_val.append(loss_val.detach().item())

        train_acc = accuracy(logits, labels, idx_train)
        val_acc = accuracy(logits, labels, idx_val)

        trace_acc_train.append(train_acc)
        trace_acc_val.append(val_acc)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if it % display_step == 0:
            logging.info(f'\nEpoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f}, '
                         f'acc_train: {train_acc:.5f}, acc_val: {val_acc:.5f} ')

    # restore the best validation state
    model.load_state_dict(best_state)
    return trace_loss_val, trace_loss_train, trace_acc_val, trace_acc_train




@ex.automain
def run(data_dir: str, dataset: str, 
        model_params: Dict[str, Any], 
        train_params: Dict[str, Any], binary_attr: bool,
        make_undirected: bool, seed: int, artifact_dir: str, model_storage_type: str, 
        # ppr_cache_params: Dict[str, str],
        device: Union[str, int], data_device: Union[str, int], display_steps: int, 
        debug_level: str
        ):
    """
    Instantiates a sacred experiment executing a training run for a given model configuration.
    Saves the model to storage and evaluates its accuracy. 

    Parameters
    ----------
    data_dir : str
        Path to data folder that contains the dataset
    dataset : str
        Name of the dataset. Either one of: `cora_ml`, `citeseer`, `pubmed` or an ogbn dataset
    model_params : Dict[str, Any], optional
        The hyperparameters of the model to be passed as keyword arguments to the constructor of the model class.
        This dict must contain the key "model" specificing the model class. Supported model classes are:
            - GCN
            - DenseGCN
            - RGCN
            - RGAT
            - PPRGo
            - RobustPPRGo
    train_params : Dict[str, Any], optional
        The training/hyperparamters to be passed as keyword arguments to the model's ".fit()" method or to 
        the global "train" method if "model.fit()" is undefined.
    device : Union[int, torch.device]
        The device to use for training. Must be `cpu` or GPU id
    data_device : Union[int, torch.device]
        The device to use for storing the dataset. For batched models (like PPRGo) this may differ from the device parameter. 
        In all other cases device takes precedence over data_device
    make_undirected : bool
        Normalizes adjacency matrix with symmetric degree normalization (non-scalable implementation!)
    binary_attr : bool
        If true the attributes are binarized (!=0)
    artifact_dir: str
        The path to the folder that acts as TinyDB Storage for trained models
    model_storage_type: str
        The name of the storage (TinyDB) table name the model is stored into.
    ppr_cache_params: Dict[str, any]
        Only used for PPRGo based models. Allows caching the ppr matrix on the hard drive and loading it from disk.
        Tthe following keys in the dictionary need be provided:
            data_artifact_dir : str
                The folder name/path in which to look for the storage (TinyDB) objects
            data_storage_type : str
                The name of the storage (TinyDB) table name that's supposed to be used for caching ppr matrices

    Returns
    -------
    Dict[str, any]
        A dictionary with the test set accuracy, the training & validation loss as well as the path to the trained model. 
    """
    if debug_level is not None and isinstance(debug_level, str):
        logger = logging.getLogger()
        if debug_level.lower() == "info":
            logger.setLevel(logging.INFO)
        if debug_level.lower() == "debug":
            logger.setLevel(logging.DEBUG)
        if debug_level.lower() == "critical":
            logger.setLevel(logging.CRITICAL)
        if debug_level.lower() == "error":
            logger.setLevel(logging.ERROR)

    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not availble, set device to 'cpu'"
        assert data_device == "cpu", "CUDA is not availble, set device to 'cpu'"

    logging.info({
        'dataset': dataset, 
        'model_params': model_params,
        'train_params': train_params, 'binary_attr': binary_attr,
        'make_undirected': make_undirected, 'seed': seed, 'artifact_dir': artifact_dir,
        'model_storage_type': model_storage_type, 
        # 'ppr_cache_params': ppr_cache_params, 
        'device': device,
        'display_steps': display_steps, 'data_device': data_device
    })

    torch.manual_seed(seed)
    np.random.seed(seed)
    graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                       binary_attr=binary_attr,
                    #    return_original_split=dataset.startswith('ogbn')
                       )

    attr, adj, labels = graph[:3]
    if len(graph) == 3 or graph[3] is None:  # TODO: This is weird
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    in_channels = attr.shape[1]
    out_channels = int(labels[~labels.isnan()].max() + 1)

    logging.info(f"Training set size: {len(idx_train)}")
    logging.info(f"Validation set size: {len(idx_val)}")
    logging.info(f"Test set size: {len(idx_test)}")

    # Collect all hyperparameters of model
    # ppr_cache = None
    # if ppr_cache_params is not None:
    #     ppr_cache = dict(ppr_cache_params)
    #     ppr_cache.update(dict(
    #         dataset=dataset,
    #         make_undirected=make_undirected,
    #     ))
    hyperparams = dict(model_params)
    hyperparams.update({
        'in_channels': in_channels,
        'out_channels': out_channels,
        'hidden_channels' : cfg.model.params.hidden_channels,
        # 'ppr_cache_params': ppr_cache,
        'train_params': train_params
    })

    model = create_model()

    logging.info("Memory Usage after loading the dataset:")
    logging.info(utils3.get_max_memory_bytes() / (1024 ** 3))

    if hasattr(model, 'fit'):
        trace = model.fit(adj, attr,
                          labels=labels,
                          idx_train=idx_train,
                          idx_val=idx_val,
                          display_step=display_steps,
                          dataset=dataset,
                          make_undirected=make_undirected,
                          **train_params)

        trace_val, trace_train = trace if trace is not None else (None, None)

    else:
        trace_val, trace_train, _, _ = train(
            model=model, attr=attr.to(device), adj=adj.to(device), labels=labels.to(device),
            idx_train=idx_train, idx_val=idx_val, display_step=display_steps, **train_params)

    model.eval()

    # For really large graphs we don't want to compute predictions for all nodes, just the test nodes is enough.
    # Calculating predictions for a sub-set of nodes is only possible for batched gnns like PPRGo
    with torch.no_grad():
        model.eval()
        # if isinstance(model, PPRGoWrapperBase):
        #     prediction = model(attr, adj, ppr_idx=idx_test)
        #     test_accuracy = (prediction.cpu().argmax(1) == labels.cpu()[idx_test]).float().mean().item()
        # else:
        prediction = model(attr, adj)
        test_accuracy = accuracy(prediction.cpu(), labels.cpu(), idx_test)

    logging.info(f'Test accuracy is {test_accuracy} with seed {seed}')

    storage = Storage(artifact_dir, experiment=ex)
    params = dict(dataset=dataset, binary_attr=binary_attr, make_undirected=make_undirected,
                  seed=seed, 
                  **hyperparams
                  )

    model_path = storage.save_model(model_storage_type, params, model)
    print('accuracy:', test_accuracy, 'trace_val:', trace_val, 'trace_train:', trace_train, 'model_path:', model_path)
    return {
        'accuracy': test_accuracy,
        'trace_val': trace_val,
        'trace_train': trace_train,
        'model_path': model_path
    }
    
    



