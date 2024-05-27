import pytest
import os
import torch
import numpy as np
from gnn_toolbox.experiment_handler.exp_gen import generate_experiments_from_yaml
from gnn_toolbox.experiment_handler.exp_runner import run_experiment
from gnn_toolbox.experiment_handler.artifact_manager import ArtifactManager
from gnn_toolbox.experiment_handler.config_validator import load_and_validate_yaml

# @pytest.fixture
# def experiment_config(tmp_path):
#     return {
#         'output_dir': str(tmp_path),
#         'cache_dir': str(tmp_path / 'cache'),
#         'experiment_templates': [
#             {
#                 'name': 'GCN_Cora_DICE_Evasion',
#                 'seed': 0,  # Use a fixed seed for reproducibility
#                 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#                 'model': {'name': 'GCN', 'params': {'hidden_channels': 64, 'dropout': 0.5}},
#                 'dataset': {'name': 'Cora', 'make_undirected': True},
#                 'attack': {
#                     'scope': 'global',
#                     'type': 'evasion',
#                     'name': 'DICE',
#                     'epsilon': [0.05, 0.1, 0.15, 0.2]  # Test with multiple epsilon values
#                 },
#                 'training': {'max_epochs': 200, 'patience': 20},
#                 'optimizer': {'name': 'Adam', 'params': {'lr': 0.01}},
#                 'loss': {'name': 'CrossEntropyLoss'}
#             }
#         ]
#     }

@pytest.fixture
def config_path():
    return os.path.join(os.path.dirname(__file__), 'test_configs', 'DICE_evasion.yaml')

@pytest.fixture
def deeprobust_results():
    # DeepRobust paper results for DICE on Cora dataset
    return {
        0.05: 0.81, # GRT result: 0.779
        0.1: 0.80, # GRT result: 0.765
        0.15: 0.78, # GRT result: 0.744
        0.2: 0.75, # GRT result: 0.708
        0.25: 0.73 # GRT result: 0.689
    }

def test_experiment_results_against_deeprobust(config_path, deeprobust_results):
    expected_clean_accuracy = 0.82 # Unperturbed clean model accuracy from DeepRobust paper
    
    experiment_config = load_and_validate_yaml(config_path)
    experiments, cache_dir = generate_experiments_from_yaml(experiment_config)
    artifact_manager = ArtifactManager(cache_dir)

    for experiment_dir, experiment in experiments.items():
        all_result, _ = run_experiment(experiment, experiment_dir, artifact_manager)

        epsilon = experiment['attack']['epsilon']
        expected_attacked_accuracy = deeprobust_results[epsilon]

        # Assert that the accuracy of GRT is close to the expected accuracy of DeepRobust to ensure the training and attacking are implemented correctly
        obtained_clean_accuracy = all_result['clean_result'][-2]['accuracy_test']
        obtained_attacked_accuracy = all_result['perturbed_result']['accuracy']
        
        assert obtained_clean_accuracy == pytest.approx(expected_clean_accuracy, abs=0.05)
        assert obtained_attacked_accuracy == pytest.approx(expected_attacked_accuracy, abs=0.05)  # Allow a tolerance of 0.05