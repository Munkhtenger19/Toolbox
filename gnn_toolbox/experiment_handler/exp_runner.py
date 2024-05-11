from gnn_toolbox.custom_modules import *
from gnn_toolbox.common import (prepare_dataset, 
                                evaluate_model,
                                train, 
                                classification_statistics,
                                gen_local_attack_nodes)
from gnn_toolbox.experiment_handler.create_modules import create_model, create_global_attack, create_local_attack, create_dataset, create_optimizer, create_loss

import torch
from torchtyping import TensorType
from typing import Union, List, Dict
from torch_sparse import SparseTensor
import numpy as np

from tqdm.auto import tqdm
import torch.nn.functional as F
import logging
from torch_geometric.seed import seed_everything
import copy
from torch.utils.tensorboard import SummaryWriter
import logging


def run_experiment(experiment, experiment_dir, artifact_manager):
    # print('experiment', experiment)
    # writer = SummaryWriter(f"{experiment_dir}")
    seed_everything(experiment['seed'])
    device = experiment['device']
    graph_index =  experiment['dataset'].pop('graph_index', None)
    make_undirected = experiment['dataset'].get('make_undirected', False)
    dataset = create_dataset(**experiment['dataset'])

    attr, adj, labels, split, num_edges = prepare_dataset(dataset, experiment, graph_index, make_undirected)
    
    model = create_model(experiment['model'])
    
    # _, accuracy = evaluate_global(model=model, attr=attr, adj=adj, labels=labels, idx_test=split['test'], device=device) 
    
    # print('untrained model accuracy', accuracy)
    
    clean_result = clean_train(experiment, artifact_manager, model, attr, adj, labels, split, device)
    
    perturbed_result = None
    if(experiment['attack']['scope'] == 'global'):
        if experiment['attack']['type'] == 'poison':
            pert_adj, pert_attr = global_attack(experiment['attack'], attr, adj, labels, split['train'], model, device, num_edges, make_undirected)
            
            model_path, perturbed_result = artifact_manager.model_exists(experiment, is_unattacked_model=False)
            
            if model_path is None or perturbed_result is None:
            #     model.load_state_dict(torch.load(model_path))
            #     model.to(device)
            #     return 
            # else:    
                perturbed_result = train_and_evaluate(model, pert_attr, pert_adj, attr, adj, labels, split, device, experiment, artifact_manager, is_unattacked_model=False)
        elif experiment['attack']['type'] == 'evasion':
            pert_adj, pert_attr = global_attack(experiment['attack'], attr, adj, labels, split['test'], model, device, num_edges, make_undirected)
            
            logits, accuracy = evaluate_model(model=model, attr=pert_attr, adj=pert_adj, labels=labels, idx_test=split['test'], device=device)
            
            perturbed_result ={
                'logits': logits.cpu().numpy().tolist(),
                'accuracy': accuracy,
            }
        
    elif(experiment['attack']['scope'] == 'local'):        
        perturbed_result = local_attack(experiment, attr, adj, labels, split, model, device, make_undirected)

    all_result = {
        'clean_result': clean_result,
        'perturbed_result': perturbed_result,
    }
    # log to tensorboard
    return all_result, experiment


def clean_train(current_config, artifact_manager, model, attr, adj, labels, split, device):
    model_path, result = artifact_manager.model_exists(current_config, is_unattacked_model=True)
    if model_path and result:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        # print('accuracy after model retrieved: ', evaluate_global(model, attr, adj, labels, split['test']), current_config.device)
        return result
    
    result = train_and_evaluate(model, attr, adj, attr, adj, labels, split, device, current_config, artifact_manager, is_unattacked_model=True)
    
    return result
    
def train_and_evaluate(model, train_attr, train_adj, test_attr, test_adj, labels, split, device, current_config, artifact_manager, is_unattacked_model):
    model = model.to(device)
    train_attr = train_attr.to(device)
    train_adj = train_adj.to(device)
    test_attr = test_attr.to(device)
    test_adj = test_adj.to(device)
    labels = labels.to(device)
    

    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    
    optimizer_params = current_config['optimizer'].get('params', {})

    optimizer = create_optimizer(current_config['optimizer']['name'], model, **optimizer_params)

    loss_params = current_config['loss'].get('params', {})
    
    loss = create_loss(current_config['loss']['name'], **loss_params)
    
    result = train(model=model, attr=train_attr, adj=train_adj, labels=labels, idx_train=split['train'], idx_val=split['valid'], idx_test=split['test'], optimizer=optimizer, loss=loss, **current_config['training'])

    _, accuracy = evaluate_model(model=model, attr=test_attr, adj=test_adj, labels=labels, idx_test=split['test'], device=device)
 
    result.append({
        'Test accuracy after best model retrieval': accuracy
    })
    # if is_unattacked_model:
    #     artifact_manager.save_model(model, current_config, result)
    # else:
    artifact_manager.save_model(model, current_config, result, is_unattacked_model)
    
    return result

def global_attack(attack_info, attr, adj, labels, idx_attack, model, device, num_edges, make_undirected):
    attack_params = getattr(attack_info, 'params', {})
    attack_model = create_global_attack(attack_info['name'])(
        attr=attr, adj=adj, labels=labels, idx_attack=idx_attack,
        model=model, device=device, make_undirected=make_undirected, **attack_params)
    
    n_perturbations = int(round(attack_info['epsilon'] * num_edges))
    attack_model.attack(n_perturbations)
    return attack_model.get_perturbations()
    

def local_attack(experiment, attr, adj, labels, split, model, device, make_undirected):
    attack_params = getattr(experiment['attack'], 'params', {})
    attack_model = create_local_attack(experiment['attack']['name'])(
        attr=attr, adj=adj, labels=labels, 
        idx_attack=split['test'],
        model=model, device=device, make_undirected=make_undirected, **attack_params)
    
    results = []
    eps = experiment['attack']['epsilon']
    # nodes = experiment.attack.nodes
    if 'nodes' not in experiment['attack']:
        epsilon_inverse = int(1 / eps)
        
        # min_node_degree = max(2, epsilon_inverse) if experiment['attack']['min_node_degree'] is None else experiment['attack']['min_node_degree']
        min_node_degree =experiment['attack'].get('min_node_degree', max(2, epsilon_inverse))
        
        # topk = int(experiment.attack.topk) if experiment.attack.topk is not None else 10

        topk = int(experiment['attack'].get('topk', 10))
    
        nodes = gen_local_attack_nodes(attr, adj, labels, model, split['train'], device, topk=topk, min_node_degree=min_node_degree)
    else:
        nodes = [int(i) for i in experiment['attack']['nodes']]
    
    for node in nodes:
        degree = adj[node].sum()
        n_perturbations = int((eps * degree).round().item())
        if n_perturbations == 0:
            logging.error(
                f"Skipping attack for model '{model}' using {experiment['attack']['name']} with eps {eps} at node {node}.")
            continue
        try:
            attack_model.attack(n_perturbations, node_idx=node)
        except Exception as e:
            logging.exception(e)
            logging.error(
                f"Failed to attack model '{model}' using {experiment['attack']['name']} with eps {eps} at node {node}.")
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
        if(experiment['attack']['type'] == 'poison'):
            perturbed_adj, perturbed_attr = attack_model.get_perturbations()
                
            victim = copy.deepcopy(model).to(device)
            for module in victim.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            
            optimizer_params = experiment['optimizer'].get('params', {})

            optimizer = create_optimizer(experiment['optimizer']['name'], model, **optimizer_params)

            loss_params = experiment['loss'].get('params', {})
            
            loss = create_loss(experiment['loss']['name'], **loss_params)
            
            _ = train(model=victim, attr=perturbed_attr.to(device), adj=perturbed_adj.to(device), labels=labels.to(device), idx_train=split['train'], idx_val=split['valid'], idx_test=split['test'], optimizer=optimizer, loss=loss, **experiment['training'])
            
            attack_model.set_eval_model(victim)
            logits_poisoning, _ = attack_model.evaluate_local(node)
            attack_model.set_eval_model(model)
            results[-1].update({
                'results after attacking (perturbed data)':
                {
                    'logits': logits_poisoning.cpu().numpy().tolist(),
                    **classification_statistics(logits_poisoning.cpu(), labels[node].long().cpu())
                },
                'pyg_margin': attack_model._probability_margin_loss(victim(attr.to(device), adj.to(device)),labels, [node]).item()
            })
    assert len(results) > 0, "No attack could be made."
    return results

def tensorboard_log(writer, results):
    for key, value in results.items():
        writer.add_scalar(key, value, results['epoch'])
