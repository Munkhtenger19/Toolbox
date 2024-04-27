from experiment import generate_experiments_from_yaml
from cmd_args import parse_args
from torch_geometric.nn import GCN
from customize.utils import create_model, create_attack
from PyG_to_sci_sparse import prep_graph
import torch
from torchtyping import TensorType
from typing import Union, List
from torch_sparse import SparseTensor
import numpy as np
from customize.data import load_dataset, register_dataset

from tqdm.auto import tqdm
import torch.nn.functional as F
import logging
from torch_geometric.seed import seed_everything
import copy
from torch.utils.tensorboard import SummaryWriter
from experiment_logger import LogExperiment

class ConfigManager:
    _global_config = {}

    @staticmethod
    def update_config(new_config):
        ConfigManager._global_config = new_config

    @staticmethod
    def get_config():
        return ConfigManager._global_config
    
def run_experiment(experiment, experiment_dir):
    writer = SummaryWriter(f"{experiment_dir}")
    ConfigManager.update_config(experiment)
    current_config = ConfigManager.get_config()
    seed_everything(current_config.seed)
    device = current_config.device
    # Setup dataset. If poisoning, convert idx_train, idx_val to desired format (DeepRobust or Robustness at scale) and attack it, convert it back to torch tensor and train the model. Don't touch the idx_test and run it against the model
    # If evasion, train the model with dataset without converting and convert  idx_test to desired format and attack it, and convert it back to torch tensor and run it against the model
    dataset = load_dataset(current_config.dataset)
    # converted adj to torch.sparse
    attr, adj, labels, split, num_edges = prep_graph(dataset, device)
    # current_config.model.params.update({
    #     'in_channels': attr.size(1),
    #     'out_channels': int(labels[~labels.isnan()].max() + 1)
    # })
    
    # add dataset info to model arguments
    current_config['model']['params'].update({
        'in_channels': attr.size(1),
        'out_channels': int(labels[~labels.isnan()].max() + 1)
    })
    
    # save_config
    
    model = create_model(current_config)
    
    
    # endees
    # attack_model = create_attack(current_config.attack.name)(attr = attr, adj = adj, labels = labels, idx_attack = split['train'], model = model, device = device, data_device= 0, make_undirected = True)
    # n_perturbations = int(round(current_config.attack.epsilon * num_edges))
    # attack_model.attack(n_perturbations)
    # pert_adj, pert_attr = attack_model.get_pertubations()
    
    # if(current_config.attack.attack_type == 'poisoning'):
        # _ = train(model=model, attr=attr.to(device), adj=adj.to(device),labels=labels.to(device), idx_train=split['train'], idx_val=split['valid'], **current_config.training)
        
        # target, accuracy = evaluate_global(model = model, attr = attr, adj = adj, labels = labels, eval_idx = split['test'])
        
        # print('========================================')
        # print('clean accuracy:', accuracy)
        
        # for module in model.modules():
        #     if hasattr(module, 'reset_parameters'):
        #         module.reset_parameters()
        
        # _ = train(model=model, attr=pert_attr.to(device), adj=pert_adj.to(device),labels=labels.to(device), idx_train=split['train'], idx_val=split['valid'], **current_config.training)
        
        # target, accuracy = evaluate_global(model = model, attr = attr, adj = adj, labels = labels, eval_idx = split['test'])
        
        # print('========================================')
        # print('perturbed accuracy:', accuracy)
        
    if(current_config.attack.attack_type == 'poison'):
        pert_adj, pert_attr = initialize_attack(current_config, attr, adj, labels, split['train'], model, device, num_edges)
        
        clean_result = train_and_evaluate(model, attr, adj, attr, adj, labels, split, device, writer, current_config.training)
        # print('========================================')
        # print('Clean accuracy:', result[-1]['accuracy_test'], clean_accuracy)
        perturbed_result = train_and_evaluate(model, pert_attr, pert_adj, attr, adj, labels, split, device, writer, current_config.training)
        # print('========================================')
        # print('Perturbed accuracy:', result[-1]['accuracy_test'], perturbed_accuracy)
        
    # elif(current_config.attack.attack_type == 'evasion'):
    #     pert_attr, pert_adj = initialize_attack(current_config, attr, adj, labels, split['test'], model, device, num_edges)
    #     clean_accuracy = train_and_evaluate(model, attr, adj, attr, adj, labels, split, device, writer, current_config.training)
    #     print('========================================')
    #     print('Clean accuracy:', clean_accuracy)
    #     perturbed_accuracy = train_and_evaluate(model, attr, adj, pert_attr, pert_adj, labels, split, device, writer, current_config.training)
    #     print('========================================')
    #     print('Perturbed accuracy:', perturbed_accuracy)
    result = {
        'clean_result': clean_result,
        'perturbed_result': perturbed_result,
    }
    return result, current_config

def train_and_evaluate(model, train_attr, train_adj, test_attr, test_adj, labels, split, device, writer, training_config):
    # Move data to the device (GPU or CPU)
    train_attr = train_attr.to(device)
    train_adj = train_adj.to(device)
    test_attr = test_attr.to(device)
    test_adj = test_adj.to(device)
    labels = labels.to(device)
    
    copied_model = copy.deepcopy(model).to(device)
    for module in copied_model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    # Train the model
    result = train(model=copied_model, attr=train_attr, adj=train_adj, labels=labels, idx_train=split['train'], idx_val=split['valid'], idx_test=split['test'], writer= writer, **training_config)

    # Evaluate the model
    _, accuracy = evaluate_global(model=copied_model, attr=test_attr, adj=test_adj, labels=labels, idx_test=split['test'])
    
    result.append({
        'Test accuracy after best model retrieval': accuracy
    })
    return result

def initialize_attack(experiment, attr, adj, labels, idx_attack, model, device, num_edges, data_device = 0, make_undirected=True):
    attack_model = create_attack(experiment.attack.name)(
        attr=attr, adj=adj, labels=labels, idx_attack=idx_attack,
        model=model, device=device, data_device=data_device, make_undirected=make_undirected
    )
    n_perturbations = int(round(experiment.attack.epsilon * num_edges))
    attack_model.attack(n_perturbations)
    return attack_model.get_perturbations()

# ! Make the optimizer custom, not only Adam
def train(model, attr, adj, labels, idx_train, idx_val, idx_test,
          lr, weight_decay, patience, max_epochs, writer,display_step=10):
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
    # trace_loss_train = []
    # trace_loss_val = []
    # trace_acc_train = []
    # trace_acc_val = []
    # trace_acc_test = []
    results = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf

    model.train()
    for epoch in tqdm(range(max_epochs), desc='Training...'):
        optimizer.zero_grad()

        logits = model(attr, adj)
        
        
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss_val = F.cross_entropy(logits[idx_val], labels[idx_val])

        loss_train.backward()
        optimizer.step()

        # trace_loss_train.append(loss_train.detach().item())
        # trace_loss_val.append(loss_val.detach().item())

        train_acc = accuracy(logits, labels, idx_train)
        val_acc = accuracy(logits, labels, idx_val)
        test_acc = accuracy(logits, labels, idx_test)
        # trace_acc_train.append(train_acc)
        # trace_acc_val.append(val_acc)
        # trace_acc_test.append(test_acc)
        
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if epoch >= best_epoch + patience:
                break

        result = {
            "epoch": epoch,
            "loss_train": loss_train.item(),
            "loss_val": loss_val.item(),
            "accuracy_train": train_acc,
            "accuracy_val": val_acc,
            "accuracy_test": test_acc,
        }
        tensorboard_log(writer, result)
        results.append(result)

    # restore the best validation state
    model.load_state_dict(best_state)
    return results

def tensorboard_log(writer, results):
    for key, value in results.items():
        writer.add_scalar(key, value, results['epoch'])

@torch.no_grad()
def evaluate_global(model,
                    attr: TensorType["n_nodes", "n_features"],
                    adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
                    labels: TensorType["n_nodes"],
                    idx_test: Union[List[int], np.ndarray]):
    """
    Evaluates any model w.r.t. accuracy for a given (perturbed) adjacency and attribute matrix.
    """
    model.eval()
    
    pred_logits_target = model(attr, adj)[idx_test]

    acc_test_target = accuracy(pred_logits_target.cpu(), labels.cpu()[idx_test], np.arange(pred_logits_target.shape[0]))

    return pred_logits_target, acc_test_target

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

if __name__ == "__main__":
    try:
        args = parse_args()
        # make main output directory
        
        experiments, experiment_dirs, output_dir = generate_experiments_from_yaml(args.cfg_file)
        
        results = []
        for experiment, curr_dir in zip(experiments, experiment_dirs):
            # make each different experiment directory
            # experiment_dir = setup_directories(output_dir, experiment['name'])
            # 
            result, experiment_cfg = run_experiment(experiment, curr_dir)
            # * here make it log the results
            LogExperiment(curr_dir, experiment_cfg, result)
        print("All experiments completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    # print('New experiment which you doing')
    # from torch_geometric.datasets import Planetoid
    # seed_everything(0)
    # device = 'cuda'
    
    # dataset = Planetoid(name = 'cora', root = './datasets')
    # attr, adj, labels, split, num_edges = prep_graph(dataset, device)
    # model = GCN(in_channels = attr.size(1), hidden_channels=32, out_channels=int(labels[~labels.isnan()].max() + 1), num_layers=2)
    # model.to(device)
    # training_params = {
    #     'max_epochs': 300,
    #     'lr': 0.001,
    #     'weight_decay': 0.0005,
    #     'patience': 300,
    # }
    # _ = train(model=model, attr=attr.to(device), adj=adj.to(device),labels=labels.to(device), idx_train=split['train'], idx_val=split['valid'],**training_params)
        
    # target, accuracy = evaluate_global(model = model, attr = attr, adj = adj, labels = labels, eval_idx = split['test'])
    # print(accuracy)