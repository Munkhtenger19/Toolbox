import torch
import logging
from typing import Any, Dict, Sequence, Union, Optional, Tuple
from attack.base_attack import Attack
from attack import create_attack
from sacred import Experiment
import numpy as np


from customize.data import load_dataset_from_cfg, set_dataset_info
import scipy.sparse as sp
from torchtyping import TensorType
from torch_sparse import SparseTensor
import torch_sparse
from attack.io import Storage
import customize.utils as utils
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

    attack = 'DICE'
    attack_params = dict(
        # epochs=500,
        # fine_tune_epochs=100,
        # keep_heuristic="WeightOnly",
        # block_size=100_000,
        # do_synchronize=True,
        # loss_type="tanhMargin",
    )
    epsilons = [0.01, 0.1]

    artifact_dir = 'cache_debug'
    model_label = "Vanilla GCN"
    model_storage_type = 'pretrained'
    pert_adj_storage_type = 'evasion_global_adj'
    pert_attr_storage_type = 'evasion_global_attr'

    debug_level = "info"

# def split(labels, n_per_class=20, seed=None):
#     """
#     Randomly split the training data.

#     Parameters
#     ----------
#     labels: array-like [num_nodes]
#         The class labels
#     n_per_class : int
#         Number of samples per class
#     seed: int
#         Seed

#     Returns
#     -------
#     split_train: array-like [n_per_class * nc]
#         The indices of the training nodes
#     split_val: array-like [n_per_class * nc]
#         The indices of the validation nodes
#     split_test array-like [num_nodes - 2*n_per_class * nc]
#         The indices of the test nodes
#     """
#     if seed is not None:
#         np.random.seed(seed)
#     nc = labels.max() + 1

#     split_train, split_val = [], []
#     for label in range(nc):
#         perm = np.random.permutation((labels == label).nonzero()[0])
#         split_train.append(perm[:n_per_class])
#         split_val.append(perm[n_per_class:2 * n_per_class])

#     split_train = np.random.permutation(np.concatenate(split_train))
#     split_val = np.random.permutation(np.concatenate(split_val))

#     assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

#     split_test = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_train, split_val)))

#     return split_train, split_val, split_test

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
    logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

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
        adj = utils.to_symmetric_scipy(adj)

        logging.debug("Memory Usage after making the graph undirected:")
        logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    logging.debug("Memory Usage after normalizing the graph")
    logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to(device)

    attr_matrix = data.x.cpu().numpy()

    attr = torch.from_numpy(attr_matrix).to(device)

    logging.debug("Memory Usage after normalizing graph attributes:")
    logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    labels = data.y.squeeze().to(device)
    # except:
    #     raise NotImplementedError(f"Dataset `with name '{name}' is not supported")

    # if binary_attr:
    #     # NOTE: do not use this for really large datasets.
    #     # The mask is a **dense** matrix of the same size as the attribute matrix
    #     attr[attr != 0] = 1
    # elif feat_norm:
    #     attr = utils.row_norm(attr)

    if split is not None:
        return attr, adj, labels, split

    return attr, adj, labels, None

def prepare_attack_experiment(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any],
                              epsilons: Sequence[float], 
                              binary_attr: bool,
                              make_undirected: bool,
                              seed: int, artifact_dir: str, pert_adj_storage_type: str, pert_attr_storage_type: str,
                              model_label: str, model_storage_type: str, device: Union[str, int],
                              surrogate_model_label: str, data_device: Union[str, int], debug_level: str,
                              ex: Experiment):

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
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'epsilons': epsilons,
        'make_undirected': make_undirected, 
        'binary_attr': binary_attr,
        'seed': seed,
        'artifact_dir':  artifact_dir, 'pert_adj_storage_type': pert_adj_storage_type,
        'pert_attr_storage_type': pert_attr_storage_type, 'model_label': model_label,
        'model_storage_type': model_storage_type, 'device': device, 'data_device': data_device
    })

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'

    # To increase consistency between runs
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                       binary_attr=binary_attr
                       )
    # data = dataset[0]

    attr, adj, labels = graph[:3]
    if graph[3] is None:
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    storage = Storage(artifact_dir, experiment=ex)

    attack_params = dict(attack_params)
    # if "ppr_cache_params" in attack_params.keys():
    #     ppr_cache_params = dict(attack_params["ppr_cache_params"])
    #     ppr_cache_params['dataset'] = dataset
    #     attack_params["ppr_cache_params"] = ppr_cache_params

    pert_params = dict(dataset=dataset,
                       binary_attr=binary_attr,
                       make_undirected=make_undirected,
                       seed=seed,
                       attack=attack,
                       model=model_label if model_label == surrogate_model_label else None,  # For legacy reasons
                       surrogate_model=surrogate_model_label,
                       attack_params=attack_params)

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr,
                        make_undirected=make_undirected,
                        seed=seed)

    # if model_label is not None and model_label:
    #     model_params["label"] = model_label

    if make_undirected:
        m = adj.nnz() / 2
    else:
        m = adj.nnz()

    return attr, adj, labels, idx_train, idx_val, idx_test, storage, attack_params, pert_params, model_params, m


def run_global_attack(epsilon, m, storage, pert_adj_storage_type, pert_attr_storage_type,
                      pert_params, adversary, model_label):
    n_perturbations = int(round(epsilon * m))

    pert_adj = storage.load_artifact(pert_adj_storage_type, {**pert_params, **{'epsilon': epsilon}})
    pert_attr = storage.load_artifact(pert_attr_storage_type, {**pert_params, **{'epsilon': epsilon}})

    if pert_adj is not None and pert_attr is not None:
        logging.info(
            f"Found cached perturbed adjacency and attribute matrix for model '{model_label}' and eps {epsilon}")
        adversary.set_pertubations(pert_adj, pert_attr)
    else:
        logging.info(f"No cached perturbations found for model '{model_label}' and eps {epsilon}. Execute attack...")
        adversary.attack(n_perturbations)
        pert_adj, pert_attr = adversary.get_pertubations()

        if n_perturbations > 0:
            storage.save_artifact(pert_adj_storage_type, {**pert_params, **{'epsilon': epsilon}}, pert_adj)
            storage.save_artifact(pert_attr_storage_type, {**pert_params, **{'epsilon': epsilon}}, pert_attr)

@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float],
        binary_attr: bool,
        make_undirected: bool, seed: int, artifact_dir: str, pert_adj_storage_type: str,
        pert_attr_storage_type: str, model_label: str, model_storage_type: str, device: Union[str, int],
        data_device: Union[str, int], debug_level: str):
    """
    Instantiates a sacred experiment executing a global direct attack run for a given model configuration.
    Caches the perturbed adjacency to storage and evaluates the models perturbed accuracy. 
    Global evasion attacks allow all nodes of the graph to be perturbed under the given budget.
    Direct attacks are used to attack a model without the use of a surrogate model.

    Parameters
    ----------
    data_dir : str
        Path to data folder that contains the dataset
    dataset : str
        Name of the dataset. Either one of: `cora_ml`, `citeseer`, `pubmed` or an ogbn dataset
    device : Union[int, torch.device]
        The device to use for training. Must be `cpu` or GPU id
    data_device : Union[int, torch.device]
        The device to use for storing the dataset. For batched models (like PPRGo) this may differ from the device parameter. 
        In all other cases device takes precedence over data_device
    make_undirected : bool
        Normalizes adjacency matrix with symmetric degree normalization (non-scalable implementation!)
    binary_attr : bool
        If true the attributes are binarized (!=0)
    attack : str
        The name of the attack class to use. Supported attacks are:
            - PRBCD
            - GreedyRBCD
            - DICE
            - FGSM
            - PGD
    attack_params : Dict[str, Any], optional
        The attack hyperparams to be passed as keyword arguments to the constructor of the attack class
    epsilons: List[float]
        The budgets for which the attack on the model should be executed.
    model_label : str, optional
        The name given to the model at train time using the experiment_train.py 
        This name is used as an identifier in combination with the dataset configuration to retrieve 
        the model to be attacked from storage. If None, all models that were fit on the given dataset 
        are attacked.
    artifact_dir: str
        The path to the folder that acts as TinyDB Storage for pretrained models
    model_storage_type: str
        The name of the storage (TinyDB) table name the model to be attacked is retrieved from.
    pert_adj_storage_type: str
        The name of the storage (TinyDB) table name the perturbed adjacency matrix is stored to
    pert_attr_storage_type: str
        The name of the storage (TinyDB) table name the perturbed attribute matrix is stored to

    Returns
    -------
    List[Dict[str, any]]
        List of result dictionaries. One for every combination of model and epsilon.
        Each result dictionary contains the model labels, epsilon value and the perturbed accuracy
    """

    results = []
    surrogate_model_label = False

    (
        attr, adj, labels, _, _, idx_test, 
        storage, \
        attack_params, pert_params, model_params, m
    ) = prepare_attack_experiment(
        data_dir, dataset, attack, attack_params, epsilons, 
        binary_attr,
        make_undirected,  seed, artifact_dir,
        pert_adj_storage_type, pert_attr_storage_type, model_label, model_storage_type, device, surrogate_model_label,
        data_device, debug_level, ex
    )

    # if model_label is not None and model_label:
    #     model_params['label'] = model_label

    models_and_hyperparams = storage.find_models(model_storage_type, model_params)
    

    for model, hyperparams in models_and_hyperparams:
        model_label = hyperparams["label"]
        # model = 
        logging.info(f"Evaluate  {attack} for model '{model_label}'.")
        adversary = create_attack(attack, attr=attr, adj=adj, labels=labels, model=model, idx_attack=idx_test,
                                  device=device, data_device=data_device, 
                                  binary_attr=binary_attr,
                                  make_undirected=make_undirected, **attack_params)

        for epsilon in epsilons:
            run_global_attack(epsilon, m, storage, pert_adj_storage_type, pert_attr_storage_type,
                              pert_params, adversary, model_label)

            adj_adversary = adversary.adj_adversary
            attr_adversary = adversary.attr_adversary

            logits, accuracy = Attack.evaluate_global(model.to(device), attr_adversary.to(device),
                                                      adj_adversary.to(device), labels, idx_test)

            results.append({
                'label': model_label,
                'epsilon': epsilon,
                'accuracy': accuracy
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    assert len(results) > 0

    return {
        'results': results
    }
    
    
