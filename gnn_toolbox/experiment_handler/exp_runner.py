import logging
import copy
from typing import List, Dict

import torch
from torch_sparse import SparseTensor
from torch_geometric.seed import seed_everything

from custom_components import *
from gnn_toolbox.common import prepare_dataset, evaluate_model, train
from gnn_toolbox.experiment_handler.create_modules import (
    create_model,
    create_global_attack,
    create_local_attack,
    create_dataset,
    create_optimizer,
    create_loss,
)
from gnn_toolbox.experiment_handler.artifact_manager import ArtifactManager

from gnn_toolbox.experiment_handler.exceptions import (
    ModelError,
    AttackError,
    GlobalAttackError,
    GlobalAttackCreationError,
    LocalAttackError,
    LocalAttackCreationError,
    ModelTrainingError,
    ModelCreationError,
    DatasetCreationError,
    DataPreparationError,
)


def run_experiment(experiment: Dict, artifact_manager: ArtifactManager):
    """Run a given experiment.

    Args:
        experiment (Dict): configuration of the experiment
        artifact_manager (ArtifactManager): instance of the ArtifactManager class

    Raises:
        DatasetCreationError: Raised when failed to instantiate the dataset
        DataPreparationError: Raised when failed to prepare the dataset
        ModelCreationError: Raised when failed to instantiate the model
        ModelTrainingError: Raised when failed to train the model
        AttackError: Raised when failed to execute the attack
        Exception: Raised when failed to execute the experiment

    Returns:
        Dict: results of the experiment
    """
    seed_everything(experiment["seed"])
    device = experiment["device"]

    try:
        dataset = create_dataset(**experiment["dataset"])
    except Exception as e:
        raise DatasetCreationError(
            f"Failed to instantiate the dataset {experiment['dataset']['name']} with the parameters."
        ) from e
    try:
        attr, adj, labels, split, num_edges = prepare_dataset(dataset, experiment)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise DataPreparationError(
            f"Failed to prepare the dataset {experiment['dataset']['name']}."
        ) from e

    try:
        model = create_model(experiment["model"])
        untrained_model_state_dict = model.state_dict()
    except Exception as e:
        raise ModelCreationError(
            f"Failed to instantiate the model {experiment['model']['name']} with the parameters."
        ) from e

    make_undirected = experiment["dataset"].get("make_undirected")
    try:
        clean_result = clean_train(
            experiment, artifact_manager, model, attr, adj, labels, split, device
        )
    except Exception as e:
        raise ModelTrainingError(
            f"Error during training the model {experiment['model']['name']} with unperturbed (clean) data"
        ) from e

    perturbed_result = None
    try:
        if experiment["attack"]["scope"] == "global":
            perturbed_result = execute_global_attack(
                experiment,
                attr,
                adj,
                labels,
                split,
                model,
                device,
                num_edges,
                make_undirected,
                untrained_model_state_dict,
                artifact_manager,
            )
        elif experiment["attack"]["scope"] == "local":
            perturbed_result = execute_local_attack(
                experiment, attr, adj, labels, split, model, device, make_undirected
            )
    except GlobalAttackError as e:
        raise AttackError(
            f"Error during global attack {experiment['attack']['name']} execution"
        ) from e
    except LocalAttackError as e:
        raise AttackError(
            f"Error during local attack {experiment['attack']['name']} execution"
        ) from e
    except Exception as e:
        raise Exception(
            f"Error during attack {experiment['attack']['name']} execution"
        ) from e
    all_result = {
        "clean_result": clean_result,
        "perturbed_result": perturbed_result if perturbed_result is not None else None,
    }

    return all_result, experiment


def execute_global_attack(
    experiment: Dict,
    attr: torch.Tensor,
    adj: SparseTensor,
    labels: torch.Tensor,
    split: Dict,
    model: torch.nn.Module,
    device: str,
    num_edges: int,
    make_undirected: bool,
    untrained_model_state_dict,
    artifact_manager: ArtifactManager,
):
    """Execute a global attack.

    Args:
        experiment (Dict): experiment configuration
        attr (torch.Tensor): node attributes or node features
        adj (SparseTensor): adjacency matrix
        labels (torch.Tensor): node labels
        split (Dict): train, validation, and test split
        model (torch.nn.Module): model
        device (str): device
        num_edges (int): number of edges
        make_undirected (bool): whether to make the graph undirected
        untrained_model_state_dict: state dictionary of the untrained model
        artifact_manager (ArtifactManager): instance of the ArtifactManager class

    Raises:
        GlobalAttackError: Raised when failed to execute the global attack

    Returns:
        List: results of the global attack
    """
    try:
        if experiment["attack"]["type"] == "poison":
            adversarial_attack, n_perturbations = instantiate_global_attack(
                experiment["attack"],
                attr,
                adj,
                labels,
                split["train"],
                model,
                device,
                num_edges,
                make_undirected,
            )
            model_path, perturbed_result = artifact_manager.model_exists(
                experiment, is_unattacked_model=False
            )

            if model_path is None or perturbed_result is None:
                try:
                    adversarial_attack.attack(n_perturbations)
                except Exception as e:
                    raise GlobalAttackError(
                        f"Error during executing 'attack' method of a global attack{experiment['attack']['name']}"
                    ) from e
                pert_adj, pert_attr = adversarial_attack.get_perturbations()
                perturbed_result = train_and_evaluate(
                    model,
                    pert_attr,
                    pert_adj,
                    attr,
                    adj,
                    labels,
                    split,
                    device,
                    experiment,
                    artifact_manager,
                    is_unattacked_model=False,
                    untrained_model_state_dict=untrained_model_state_dict,
                )
        elif experiment["attack"]["type"] == "evasion":
            adversarial_attack, n_perturbations = instantiate_global_attack(
                experiment["attack"],
                attr,
                adj,
                labels,
                split["test"],
                model,
                device,
                num_edges,
                make_undirected,
            )
            adversarial_attack.attack(n_perturbations)
            pert_adj, pert_attr = adversarial_attack.get_perturbations()
            logits, accuracy = evaluate_model(
                model=model,
                attr=pert_attr,
                adj=pert_adj,
                labels=labels,
                idx_test=split["test"],
                device=device,
            )
            perturbed_result = {
                "accuracy of the model": accuracy,
                "logits (raw predictions generated by model) ": logits.cpu()
                .numpy()
                .tolist(),
            }
        return perturbed_result
    except GlobalAttackCreationError as e:
        raise GlobalAttackError(
            f"Error during global attack {experiment['attack']['name']} instantiation"
        ) from e
    except Exception as e:
        raise GlobalAttackError(
            f"Error during global attack {experiment['attack']['name']} execution"
        ) from e


def clean_train(
    current_config: Dict,
    artifact_manager: ArtifactManager,
    model: torch.nn.Module,
    attr: torch.Tensor,
    adj: SparseTensor,
    labels: torch.Tensor,
    split: Dict,
    device: str,
):
    """Train the model with clean data.

    Args:
        current_config (Dict): experiment configuration
        artifact_manager (ArtifactManager): artifact manager instance
        model (torch.nn.Module): model
        attr (torch.Tensor): node attribute
        adj (SparseTensor): adjacency matrix
        labels (torch.Tensor): node labels
        split (Dict): training, validation, and test split in dictionary
        device (str): device

    """
    model_path, result = artifact_manager.model_exists(
        current_config, is_unattacked_model=True
    )
    if model_path and result:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return result

    result = train_and_evaluate(
        model,
        attr,
        adj,
        attr,
        adj,
        labels,
        split,
        device,
        current_config,
        artifact_manager,
        is_unattacked_model=True,
    )

    return result


def train_and_evaluate(
    model: torch.nn.Module,
    train_attr: torch.Tensor,
    train_adj: SparseTensor,
    test_attr: torch.Tensor,
    test_adj: SparseTensor,
    labels: torch.Tensor,
    split: Dict,
    device: str,
    current_config: Dict,
    artifact_manager: ArtifactManager,
    is_unattacked_model: bool,
    untrained_model_state_dict=None,
):
    """Train the model and evaluate it.

    Args:
        model (torch.nn.Module): Model
        train_attr (torch.Tensor): Training node attributes
        train_adj (SparseTensor): Training adjacency matrix
        test_attr (torch.Tensor): Test node attributes
        test_adj (SparseTensor): Test adjacency matrix
        labels (torch.Tensor): Node labels
        split (Dict): Train, validation, and test split
        device (str): Device
        current_config (Dict): Experiment configuration
        artifact_manager (ArtifactManager): Artifact manager instance
        is_unattacked_model (bool): Whether the model is unattacked
        untrained_model_state_dict (_type_, optional): untrained model state dict. Defaults to None.

    Returns:
        List: Results of the model training and evaluation
    """
    model = model.to(device)
    train_attr = train_attr.to(device)
    train_adj = train_adj.to(device)
    test_attr = test_attr.to(device)
    test_adj = test_adj.to(device)
    labels = labels.to(device)

    if untrained_model_state_dict is not None:
        model.load_state_dict(untrained_model_state_dict)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    optimizer_params = current_config["optimizer"].get("params", {})

    optimizer = create_optimizer(
        current_config["optimizer"]["name"], model, **optimizer_params
    )

    loss_params = current_config["loss"].get("params", {})

    loss = create_loss(current_config["loss"]["name"], **loss_params)

    result = train(
        model=model,
        attr=train_attr,
        adj=train_adj,
        labels=labels,
        idx_train=split["train"],
        idx_val=split["valid"],
        idx_test=split["test"],
        optimizer=optimizer,
        loss=loss,
        **current_config["training"],
    )

    _, accuracy = evaluate_model(
        model=model,
        attr=test_attr,
        adj=test_adj,
        labels=labels,
        idx_test=split["test"],
        device=device,
    )

    result.append({"Test accuracy after the best model retrieval": accuracy})

    artifact_manager.save_model(model, current_config, result, is_unattacked_model)

    return result


def instantiate_global_attack(
    attack_info: Dict,
    attr: torch.Tensor,
    adj: SparseTensor,
    labels: torch.Tensor,
    idx_attack: List,
    model: torch.nn.Module,
    device: str,
    num_edges: int,
    make_undirected: bool,
):
    """instantiate a global adversarial attack.

    Args:
        attack_info (Dict): Attack configuration
        attr (torch.Tensor): node attributes or node features
        adj (SparseTensor): adjacency matrix
        labels (torch.Tensor): node labels
        idx_attack (List): list of node indices to attack
        model (torch.nn.Module): model
        device (str): device
        num_edges (int): number of edges
        make_undirected (bool): whether to make the graph undirected

    Raises:
        GlobalAttackCreationError: Raised when failed to instantiate the global attack

    Returns:
        attack model and number of perturbations
    """
    attack_params = getattr(attack_info, "params", {})
    try:
        attack_model = create_global_attack(attack_info["name"])(
            attr=attr,
            adj=adj,
            labels=labels,
            idx_attack=idx_attack,
            model=model,
            device=device,
            make_undirected=make_undirected,
            **attack_params,
        )
        n_perturbations = int(round(attack_info["epsilon"] * num_edges))
        return attack_model, n_perturbations
    except Exception as e:
        raise GlobalAttackCreationError(
            f"Failed to create global adversarial attack '{attack_info['name']}'."
        ) from e


def execute_local_attack(
    experiment: Dict,
    attr: torch.Tensor,
    adj: SparseTensor,
    labels: torch.Tensor,
    split: Dict,
    model: torch.nn.Module,
    device: str,
    make_undirected: bool,
):
    """Execute a local adversarial attack.

    Args:
        experiment (Dict): experiment configuration
        attr (torch.Tensor): node attributes or node features
        adj (SparseTensor): adjacency matrix
        labels (torch.Tensor): node labels
        split (Dict): dictionary of train, validation, and test split
        model (torch.nn.Module): model
        device (str): device
        make_undirected (bool): whether to make the graph undirected

    Raises:
        LocalAttackCreationError: Raised when failed to instantiate the local attack
        LocalAttackError: Raised when failed to execute the local attack

    Returns:
        Dict: attack result
    """
    attack_params = getattr(experiment["attack"], "params", {})
    try:
        attack_model = create_local_attack(experiment["attack"]["name"])(
            attr=attr,
            adj=adj,
            labels=labels,
            idx_attack=split["test"],
            model=model,
            device=device,
            make_undirected=make_undirected,
            **attack_params,
        )
    except Exception as e:
        raise LocalAttackCreationError(
            f"Failed to create local adversarial attack '{experiment['attack']['name']}'."
        ) from e

    results = []
    eps = experiment["attack"]["epsilon"]
    try:
        nodes = [int(i) for i in experiment["attack"]["nodes"]]

        for node in nodes:
            degree = adj[node].sum()
            n_perturbations = int((eps * degree).round().item())
            if n_perturbations == 0:
                logging.error(
                    f"Number of perturbations is 0 for model {experiment['model']['name']} using {experiment['attack']['name']} with eps {eps} at node {node}. Skipping the attack to node {node}"
                )
                continue
            try:
                attack_model.attack(n_perturbations, node_idx=node)
            except Exception as e:
                raise LocalAttackError(
                    f"Error during executing 'attack' method of a global attack{experiment['attack']['name']} using with eps {eps} at node {node}."
                ) from e

            logits, initial_logits = attack_model.evaluate_node(node)

            results.append(
                {
                    "node index": node,
                    "node degree": int(degree.item()),
                    "number of perturbations": n_perturbations,
                    "target": labels[node].item(),
                    "perturbed_edges": attack_model.get_perturbed_edges()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "results before attacking (unperturbed data)": {
                        "logits": initial_logits.cpu().numpy().tolist(),
                        **attack_model.classification_statistics(
                            initial_logits.cpu(), labels[node].long().cpu()
                        ),
                    },
                    "results after attacking (perturbed data)": {
                        "logits": logits.cpu().numpy().tolist(),
                        **attack_model.classification_statistics(
                            logits.cpu(), labels[node].long().cpu()
                        ),
                    },
                }
            )
            if experiment["attack"]["type"] == "poison":
                perturbed_adj, perturbed_attr = attack_model.get_perturbations()

                victim = copy.deepcopy(model).to(device)
                for module in victim.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()

                optimizer_params = experiment["optimizer"].get("params", {})

                optimizer = create_optimizer(
                    experiment["optimizer"]["name"], model, **optimizer_params
                )

                loss_params = experiment["loss"].get("params", {})

                loss = create_loss(experiment["loss"]["name"], **loss_params)

                _ = train(
                    model=victim,
                    attr=perturbed_attr.to(device),
                    adj=perturbed_adj.to(device),
                    labels=labels.to(device),
                    idx_train=split["train"],
                    idx_val=split["valid"],
                    idx_test=split["test"],
                    optimizer=optimizer,
                    loss=loss,
                    **experiment["training"],
                )

                attack_model.set_eval_model(victim)
                logits_poisoning, _ = attack_model.evaluate_node(node)
                attack_model.set_eval_model(model)
                results[-1].update(
                    {
                        "results after attacking (perturbed data)": {
                            "logits": logits_poisoning.cpu().numpy().tolist(),
                            **attack_model.classification_statistics(
                                logits_poisoning.cpu(), labels[node].long().cpu()
                            ),
                        }
                    }
                )
            logging.info(
                f'Node {node} with perturbed edges evaluated on model {experiment["model"]["name"]} using adversarial attack {experiment["attack"]["name"]} with epsilon {eps}'
            )
            logging.debug(results[-1])
    except LocalAttackCreationError as e:
        raise LocalAttackError(
            f"Error during local attack {experiment['attack']['name']} instantiation"
        ) from e
    except Exception as e:
        raise LocalAttackError(
            f"Error during local attack {experiment['attack']['name']} execution"
        ) from e
    assert len(results) > 0, "No attack could be made."
    return results
