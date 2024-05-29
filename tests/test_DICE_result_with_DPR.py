import pytest
import os

from gnn_toolbox.experiment_handler.exp_gen import generate_experiments_from_yaml
from gnn_toolbox.experiment_handler.exp_runner import run_experiment
from gnn_toolbox.experiment_handler.artifact_manager import ArtifactManager
from gnn_toolbox.experiment_handler.config_validator import load_and_validate_yaml

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
        all_result, _ = run_experiment(experiment, artifact_manager)

        epsilon = experiment['attack']['epsilon']
        expected_attacked_accuracy = deeprobust_results[epsilon]

        # Assert that the accuracy of GRT is close to the expected accuracy of DeepRobust to ensure the training and attacking are implemented correctly
        obtained_clean_accuracy = all_result['clean_result'][-2]['accuracy_test']
        obtained_attacked_accuracy = all_result['perturbed_result']['accuracy of the model']
        
        assert obtained_clean_accuracy == pytest.approx(expected_clean_accuracy, abs=0.05)
        assert obtained_attacked_accuracy == pytest.approx(expected_attacked_accuracy, abs=0.05)  # Allow a tolerance of 0.05