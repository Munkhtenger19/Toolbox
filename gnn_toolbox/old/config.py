from gnn_toolbox.experiment_handler.exp_gen import generate_experiments_from_yaml
from cmd_args import parse_args
from torch_geometric.nn import GCN
from gnn_toolbox.old.utils3 import create_model, create_attack
from PyG_to_sci_sparse import prep_graph
import torch
from torchtyping import TensorType
from typing import Union, List, Dict
from torch_sparse import SparseTensor
import numpy as np
from custom_modules.data import load_dataset, register_dataset

from tqdm.auto import tqdm
import torch.nn.functional as F
import logging
from torch_geometric.seed import seed_everything
import copy
from torch.utils.tensorboard import SummaryWriter
from gnn_toolbox.exp_logger import LogExperiment
from gnn_toolbox.artifact_manager import ArtifactManager
import logging

class ConfigManager:
    _global_config = {}

    @staticmethod
    def update_config(new_config):
        ConfigManager._global_config = new_config

    @staticmethod
    def get_config():
        return ConfigManager._global_config
    
def run_experiment(experiment, experiment_dir, artifact_manager):
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
    
    _, accuracy = evaluate_global(model=model, attr=attr, adj=adj, labels=labels, idx_test=split['test'], device=device) 
    
    print('untrained model accuracy', accuracy)
    
    clean_result = clean_train(current_config, artifact_manager, model, attr, adj, labels, split, device, writer)
    
    _, accuracy = evaluate_global(model=model, attr=attr, adj=adj, labels=labels, idx_test=split['test'], device=device) 
    
    print('trained model accuracy', accuracy)
    # clean_result = train_and_evaluate(model, attr, adj, attr, adj, labels, split, device, writer, current_config.training, current_config.optimizer.params)
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
    perturbed_result = None
    if(current_config.attack.scope == 'global'):
        # poisoning
        if current_config.attack.type == 'poison':
            pert_adj, pert_attr = global_attack(current_config, attr, adj, labels, split['train'], model, device, num_edges)
            
            perturbed_result = train_and_evaluate(model, pert_attr, pert_adj, attr, adj, labels, split, device, writer, current_config, retrain=True, is_unattacked_model=False)
        elif current_config.attack.type == 'evasion':
            pert_adj, pert_attr = global_attack(current_config, attr, adj, labels, split['test'], model, device, num_edges)
            
            logits, accuracy = evaluate_global(model=model, attr=pert_attr, adj=pert_adj, labels=labels, idx_test=split['test'], device=device)
            
            perturbed_result ={
                'logits': logits.cpu().numpy().tolist(),
                'accuracy': accuracy,
            }
            
        # print(pert_adj)
        # print('========================================')
        # print('Clean accuracy:', result[-1]['accuracy_test'], clean_accuracy)
        
        
        
        # print('========================================')
        # print('Perturbed accuracy:', result[-1]['accuracy_test'], perturbed_accuracy)
        
    elif(current_config.attack.scope == 'local'):
        # * use the clean train model and check against pertubed test
        
        perturbed_result = local_attack(current_config, attr, adj, labels, split, model, device, num_edges)
    #     clean_accuracy = train_and_evaluate(model, attr, adj, attr, adj, labels, split, device, writer, current_config.training)
    #     print('========================================')
    #     print('Clean accuracy:', clean_accuracy)
    #     perturbed_accuracy = train_and_evaluate(model, attr, adj, pert_attr, pert_adj, labels, split, device, writer, current_config.training)
    #     print('========================================')
    #     print('Perturbed accuracy:', perturbed_accuracy)
    all_result = {
        'clean_result': clean_result,
        'perturbed_result': perturbed_result,
    }
    # log to tensorboard
    return all_result, current_config

def clean_train(current_config, artifact_manager, model, attr, adj, labels, split, device, writer):
    model_path, result = artifact_manager.model_exists(current_config)
    if model_path:
        print('found model2')
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        # now transfer the summary writer to the another one
        # no need for acc
        # print('accuracy after model retrieved: ', evaluate_global(model, attr, adj, labels, split['test']), current_config.device)
        return result
    
    result = train_and_evaluate(model, attr, adj, attr, adj, labels, split, device, writer, current_config, retrain=False, is_unattacked_model=True)
    
    return result
    
def train_and_evaluate(model, train_attr, train_adj, test_attr, test_adj, labels, split, device, writer, current_config, retrain, is_unattacked_model):
    # Move data to the device (GPU or CPU)
    train_attr = train_attr.to(device)
    train_adj = train_adj.to(device)
    test_attr = test_attr.to(device)
    test_adj = test_adj.to(device)
    labels = labels.to(device)
    
    if(retrain):
        model = copy.deepcopy(model).to(device)
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    # check for the model        
    
    # * make copy of optimizer
    
    
    # Train the model
    result = train(model=model, attr=train_attr, adj=train_adj, labels=labels, idx_train=split['train'], idx_val=split['valid'], idx_test=split['test'], writer= writer, **current_config.training, **current_config.optimizer.params)

    # Evaluate the model
    _, accuracy = evaluate_global(model=model, attr=test_attr, adj=test_adj, labels=labels, idx_test=split['test'], device=device)
    
    
        
    result.append({
        'Test accuracy after best model retrieval': accuracy
    })

    artifact_manager.save_model(model, current_config, result, is_unattacked_model)
    
    return result

def global_attack(experiment, attr, adj, labels, idx_attack, model, device, num_edges, data_device = 0, make_undirected=True):
    attack_model = create_attack(experiment.attack.name)(
        attr=attr, adj=adj, labels=labels, idx_attack=idx_attack,
        model=model, device=device, data_device=data_device, 
        **getattr(experiment.attack, 'params', {})
    )
    
    n_perturbations = int(round(experiment.attack.epsilon * num_edges))
    attack_model.attack(n_perturbations)
    return attack_model.get_perturbations()
    # n_perturbations = int(round(experiment.attack.epsilon * num_edges))
    
    # if(experiment.attack.scope == 'local' and experiment.attack.type =='poison'):
    #     try:
    #         nodes = [int(i) for i in experiment.attack.nodes]
    #     except KeyError:
    #         nodes = get_local_attack_nodes(attr, adj, labels, model, idx_attack, device, attack_type=experiment.attack.type, topk=int(experiment.attack.nodes_topk), min_node_degree=int(1/experiment.attack.epsilons))
    #     for node in nodes:
    #         attack_model.attack(node, n_perturbations)
    
    # attack_model.attack(n_perturbations)
    # return attack_model.get_perturbations()

    # * if poison, train the model on the clean data, get the nodes if they are not given. Then I need to perturb the node and train on it
    # * idx_attack is just for global attack
    
    # if(experiment.attack.scope == 'local' and experiment.attack.type =='poison'):
    #     results = []
    #     eps = experiment.attack.epsilon
    #     nodes = experiment.attack.nodes
    #     if nodes is None:
    #         epsilon_inverse = int(1 / eps)
            
    #         min_node_degree = max(2, epsilon_inverse) if experiment.attack.min_node_degree is None else experiment.attack.min_node_degree
            
    #         topk = int(experiment.attack.topk) if experiment.attack.topk is not None else 10

    #         nodes = get_local_attack_nodes(attr, adj, labels, model, split['train'], device, topk=topk, min_node_degree=min_node_degree)

    #     nodes = [int(i) for i in nodes]
    #     for node in nodes:
    #         degree = adj[node].sum()
    #         n_perturbations = int((eps * degree).round().item())
    #         if n_perturbations == 0:
    #             print(
    #                 f"Skipping attack for model '{model}' using {experiment.attack.name} with eps {eps} at node {node}.")
    #             continue
    #         try:
    #             attack_model.attack(n_perturbations, node_idx=node)
    #         except Exception as e:
    #             logging.exception(e)
    #             logging.error(
    #                 f"Failed to attack model '{model}' using {experiment.attack.name} with eps {eps} at node {node}.")
    #             continue
            
    #         logits, initial_logits = attack_model.evaluate_local(node)
            
    #         results.append({
    #             'node index': node,
    #             'node degree': int(degree.item()),
    #             'number of perturbations': n_perturbations,
                
    #             'target': labels[node].item(),                
    #             'perturbed_edges': attack_model.get_perturbed_edges().cpu().numpy().tolist(),
    #             'results before attacking (unperturbed data)': {
    #                 'logits': initial_logits.cpu().numpy().tolist(),
    #                 **classification_statistics(initial_logits.cpu(), labels[node].long().cpu())
    #             },
    #             'results after attacking (perturbed data)': {
    #                 'logits': logits.cpu().numpy().tolist(),
    #                 **classification_statistics(logits.cpu(), labels[node].long().cpu())
    #             }
    #         })
            
    #         # odoo perturbed data deeree train hiiged result avna. Deer bolhor trained model deerees perturbed data avj bna
            
    #         # if it is poison, give the attack trained model to get the perturbed adj, perturbed feature and train on it and see if it incorrectly classifies the node
    #         if experiment.attack.type == 'poison':
    #             perturbed_adj, perturbed_attr = attack_model.get_perturbations()
                
    #             victim = copy.deepcopy(model).to(device)
    #             for module in victim.modules():
    #                 if hasattr(module, 'reset_parameters'):
    #                     module.reset_parameters()
                
    #             _ = train(model=victim, attr=perturbed_attr.to(device), adj=perturbed_adj.to(device), labels=labels.to(device), idx_train=split['train'], idx_val=split['valid'], idx_test=split['test'], **experiment.training, **experiment.optimizer.params)
                
    #             # hervee eniig attacked_model-oor solichihvol 
    #             attack_model.set_eval_model(victim)
    #             logits_poisoning, _ = attack_model.evaluate_local(node)
    #             attack_model.set_eval_model(model)
                
    #             results.append({
    #                 'attacked node': node,
    #                 'logits': logits_poisoning.cpu().numpy().tolist(),
    #                 **attack_model.classification_statistics(
    #                     logits_poisoning, labels[node].long()),
                    
    #                 # 'node': node,
    #                 # 'degree': degree.item(),
    #                 # 'n_perturbations': n_perturbations,
    #                 # 'initial_logits': initial_logits,
    #                 # 'logits_poisoning': logits_poisoning,
    #             })
            
    #         results.append({
    #             'pyg_margin': attack_model._probability_margin_loss(victim(attr.to(device), adj.to(device)),labels, [node]).item()
    #         })
            
    # results.append({
    #     'all attacked nodes': nodes
    # })
    # print('results', results)
    # return results, None

def local_attack(experiment, attr, adj, labels, split, model, device, num_edges, data_device=0):
    attack_model = create_attack(experiment.attack.name)(
        attr=attr, adj=adj, labels=labels, 
        idx_attack=split['test'],
        model=model, device=device, data_device=data_device, 
        **getattr(experiment.attack, 'params', {})
    )
    
    results = []
    eps = experiment.attack.epsilon
    nodes = experiment.attack.nodes
    if nodes is None:
        epsilon_inverse = int(1 / eps)
        
        min_node_degree = max(2, epsilon_inverse) if experiment.attack.min_node_degree is None else experiment.attack.min_node_degree
        
        topk = int(experiment.attack.topk) if experiment.attack.topk is not None else 10

        nodes = get_local_attack_nodes(attr, adj, labels, model, split['train'], device, topk=topk, min_node_degree=min_node_degree)

    nodes = [int(i) for i in nodes]
    for node in nodes:
        degree = adj[node].sum()
        n_perturbations = int((eps * degree).round().item())
        if n_perturbations == 0:
            print(
                f"Skipping attack for model '{model}' using {experiment.attack.name} with eps {eps} at node {node}.")
            continue
        try:
            attack_model.attack(n_perturbations, node_idx=node)
        except Exception as e:
            logging.exception(e)
            logging.error(
                f"Failed to attack model '{model}' using {experiment.attack.name} with eps {eps} at node {node}.")
            continue
        
        logits, initial_logits = attack_model.evaluate_local(node)
  
        results.append({
            'node index': node,
            'node degree': int(degree.item()),
            'number of perturbations': n_perturbations,
            'target': labels[node].item(),                
            'perturbed_edges': attack_model.get_perturbed_edges().cpu().numpy().tolist(),
            'results before attacking (unperturbed data)': {
                'logits': initial_logits.cpu().numpy().tolist(),
                **classification_statistics(initial_logits.cpu(), labels[node].long().cpu())
            },
            'results after attacking (perturbed data)': {
                'logits': logits.cpu().numpy().tolist(),
                **classification_statistics(logits.cpu(), labels[node].long().cpu())
            }
        })
        if(experiment.attack.type == 'poison'):
            perturbed_adj, perturbed_attr = attack_model.get_perturbations()
                
            victim = copy.deepcopy(model).to(device)
            for module in victim.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            
            _ = train(model=victim, attr=perturbed_attr.to(device), adj=perturbed_adj.to(device), labels=labels.to(device), idx_train=split['train'], idx_val=split['valid'], idx_test=split['test'], **experiment.training, **experiment.optimizer.params)
            
            # hervee eniig attacked_model-oor solichihvol 
            attack_model.set_eval_model(victim)
            logits_poisoning, _ = attack_model.evaluate_local(node)
            attack_model.set_eval_model(model)
            results[-1].update({
                'results after attacking (perturbed data)':
                {
                    'logits': logits_poisoning.cpu().numpy().tolist(),
                    **classification_statistics(logits_poisoning.cpu(), labels[node].long().cpu())
                },
            })
    assert len(results) > 0, "No attack could be made."
    return results
            

def classification_statistics(logits: TensorType[1, "n_classes"],
                                  label: TensorType[()]) -> Dict[str, float]:
    logits, label = F.log_softmax(logits.cpu(), dim=-1), label.cpu()
    logits = logits[0]
    logit_target = logits[label].item()
    sorted = logits.argsort()
    logit_best_non_target = (logits[sorted[sorted != label][-1]]).item()
    confidence_target = np.exp(logit_target)
    confidence_non_target = np.exp(logit_best_non_target)
    margin = confidence_target - confidence_non_target
    return {
        'logit_target': logit_target,
        'logit_best_non_target': logit_best_non_target,
        'confidence_target': confidence_target,
        'confidence_non_target': confidence_non_target,
        'margin': margin
    }

def get_local_attack_nodes(attr, adj, labels, model, idx_test, device, topk=10, min_node_degree=2):
    # if attack_type == 'poison':
    #     model = model.to(device)
    #     train(model, attr, adj, labels, idx_test, idx_test, idx_test, 0.01, 0.0005, 300, 300, None, 10)
        
    # elif attack_type == 'evasion':
    #     with torch.no_grad():
    #         model = model.to(device)
    #         model.eval()

    #         logits = model(attr.to(device), adj.to(device))[idx_test]

    #         acc = accuracy(logits.cpu(), labels.cpu()[idx_test], np.arange(logits.shape[0]))

    logits, acc = evaluate_global(model, attr, adj, labels, idx_test, device)
    
    logging.info(f"Sample Attack Nodes for model with accuracy {acc:.4}")

    max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx = sample_attack_nodes(
        logits, labels[idx_test], idx_test, adj, topk,  min_node_degree)
    tmp_nodes = np.concatenate((max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx))
    logging.info(
        f"Sample the following attack nodes:\n{max_confidence_nodes_idx}\n{min_confidence_nodes_idx}\n{rand_nodes_idx}")
    return tmp_nodes

def sample_attack_nodes(logits: torch.Tensor, labels: torch.Tensor, nodes_idx,
                        adj: SparseTensor, topk: int, min_node_degree: int):
    assert logits.shape[0] == labels.shape[0]
    if isinstance(nodes_idx, torch.Tensor):
        nodes_idx = nodes_idx.cpu()
    node_degrees = adj[nodes_idx.tolist()].sum(-1)
    print('len(node_degrees)', len(node_degrees))
    suitable_nodes_mask = (node_degrees >= min_node_degree).cpu()

    labels = labels.cpu()[suitable_nodes_mask]
    confidences = F.softmax(logits.cpu()[suitable_nodes_mask], dim=-1)

    correctly_classifed = confidences.max(-1).indices == labels

    logging.info(
        f"Found {sum(suitable_nodes_mask)} suitable '{min_node_degree}+ degree' nodes out of {len(nodes_idx)} "
        f"candidate nodes to be sampled from for the attack of which {correctly_classifed.sum().item()} have the "
        "correct class label")
    print(
        f"Found {sum(suitable_nodes_mask)} suitable '{min_node_degree}+ degree' nodes out of {len(nodes_idx)} "
        f"candidate nodes to be sampled from for the attack of which {correctly_classifed.sum().item()} have the "
        "correct class label")
    print(sum(suitable_nodes_mask))
    assert sum(suitable_nodes_mask) >= (topk * 4), \
        f"There are not enough suitable nodes to sample {(topk*4)} nodes from"

    _, max_confidence_nodes_idx = torch.topk(confidences[correctly_classifed].max(-1).values, k=topk)
    _, min_confidence_nodes_idx = torch.topk(-confidences[correctly_classifed].max(-1).values, k=topk)

    rand_nodes_idx = np.arange(correctly_classifed.sum().item())
    rand_nodes_idx = np.setdiff1d(rand_nodes_idx, max_confidence_nodes_idx)
    rand_nodes_idx = np.setdiff1d(rand_nodes_idx, min_confidence_nodes_idx)
    rnd_sample_size = min((topk * 2), len(rand_nodes_idx))
    rand_nodes_idx = np.random.choice(rand_nodes_idx, size=rnd_sample_size, replace=False)

    return (np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][max_confidence_nodes_idx])[None].flatten(),
            np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][min_confidence_nodes_idx])[None].flatten(),
            np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][rand_nodes_idx])[None].flatten())


# ! Make the optimizer custom, not only Adam
def train(model, attr, adj, labels, idx_train, idx_val, idx_test, lr, weight_decay, patience, max_epochs, writer=None, display_step=10):
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

        # * we can run through the different metrics and append them to the results array
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
        if writer:
            tensorboard_log(writer, result)
        results.append(result)

    # restore the best validation state
    model.load_state_dict(best_state)
    # save the model
    
    return results

def tensorboard_log(writer, results):
    for key, value in results.items():
        writer.add_scalar(key, value, results['epoch'])

@torch.no_grad()
def evaluate_global(model,
                    attr: TensorType["n_nodes", "n_features"],
                    adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
                    labels: TensorType["n_nodes"],
                    idx_test: Union[List[int], np.ndarray],
                    device):
    """
    Evaluates any model w.r.t. accuracy for a given (perturbed) adjacency and attribute matrix.
    """
    model = model.to(device)
    model.eval()
    
    pred_logits_target = model(attr.to(device), adj.to(device))[idx_test]

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

    args = parse_args()
    # make it work for multiple yaml files
    
    experiments, experiment_dirs, output_dir = generate_experiments_from_yaml(args.cfg_file)
    print(f"Running {len(experiments)} experiments.")
    results = []
    # * cache here is hardcoded, should be given in each file
    artifact_manager = ArtifactManager('cache')
    for experiment, curr_dir in zip(experiments, experiment_dirs):
        
        # make each different experiment directory
        # experiment_dir = setup_directories(output_dir, experiment['name'])
        # 
        result, experiment_cfg = run_experiment(experiment, curr_dir, artifact_manager)
        # * here make it log the results
        LogExperiment(curr_dir, experiment_cfg, result)
    print("All experiments completed successfully.")

        
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