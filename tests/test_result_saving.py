import pytest
import json
import shutil
from unittest.mock import patch
from gnn_toolbox.experiment_handler.result_saver import LogExperiment
from pathlib import Path


@pytest.fixture
def experiment_dir(tmp_path):
    result_dir = tmp_path / "experiment_results"
    result_dir.mkdir(exist_ok=True)
    yield result_dir
    shutil.rmtree(result_dir)

@pytest.fixture
def experiment_cfg_evasion():
    return {
        'name': 'TestExperiment',
        'seed': ['0', '1'],
        'model': {
            'name': ['GCN', 'GCN2'],
            'params': {
                'hidden_channels': '64'
            }
        },
        'dataset': {
            'name': ['Cora', 'CiteSeer', 'PubMed'],
            'root': './datasets',
        },
        'attack': {
            'scope': 'global',
            'type': 'evasion', # Evasion attack to save result as only json
            'name': 'DICE',
            'epsilon': ['0.1', '0.2']
        },
        'training': {
            'max_epochs': '100',
            'patience': '80'
        },
        'optimizer': {
            'name': ['Adam', 'sgd'],
            'params': {
                'lr': '0.01',
                'weight_decay': '0.0005'
            }
        },
        'loss': {
            'name': 'CE'
        }
    }
    
@pytest.fixture
def experiment_cfg_poison():
    return {
        'name': 'TestExperiment',
        'seed': ['0', '1'],
        'model': {
            'name': ['GCN', 'GCN2'],
            'params': {
                'hidden_channels': '64'
            }
        },
        'dataset': {
            'name': ['Cora', 'CiteSeer', 'PubMed'],
            'root': './datasets',
        },
        'attack': {
            'scope': 'global',
            'type': 'poison', # Poison attack to save result as json and csv
            'name': 'PRBCD', 
            'epsilon': ['0.1', '0.2']
        },
        'training': {
            'max_epochs': '100',
            'patience': '80'
        },
        'optimizer': {
            'name': ['Adam', 'sgd'],
            'params': {
                'lr': '0.01',
                'weight_decay': '0.0005'
            }
        },
        'loss': {
            'name': 'CE'
        }
    }

@pytest.fixture
def result():
    return {
        'clean_result': [
            {'epoch': 0, 'loss': 1.23, 'accuracy': 0.75},
            {'epoch': 1, 'loss': 0.98, 'accuracy': 0.82},
            {'epoch': 2, 'loss': 0.85, 'accuracy': 0.88},
            {'epoch': 3, 'loss': 0.79, 'accuracy': 0.91},
            {'epoch': 4, 'loss': 0.72, 'accuracy': 0.94},
            {'epoch': 5, 'loss': 0.68, 'accuracy': 0.96}
        ],
        'perturbed_result': [
            {'epoch': 0, 'loss': 1.42, 'accuracy': 0.64},
            {'epoch': 1, 'loss': 1.26, 'accuracy': 0.71},
            {'epoch': 2, 'loss': 1.18, 'accuracy': 0.78},
            {'epoch': 3, 'loss': 1.12, 'accuracy': 0.82},
            {'epoch': 4, 'loss': 1.07, 'accuracy': 0.86},
            {'epoch': 5, 'loss': 1.03, 'accuracy': 0.89}
        ]
    }

def test_save_results_with_csv_poison(experiment_dir, experiment_cfg_poison, result):
    experiment_logger = LogExperiment(experiment_dir, experiment_cfg_poison, result, csv_save=True)
    experiment_logger.save_results()

    assert Path(experiment_dir / 'experiment_config.json').exists()
    assert Path(experiment_dir / 'clean_result.json').exists()
    assert Path(experiment_dir / 'clean_result.csv').exists()
    assert Path(experiment_dir / 'perturbed_result.json').exists()
    assert Path(experiment_dir / 'perturbed_result.csv').exists()

    with open(experiment_dir / 'experiment_config.json', 'r') as f:
        loaded_config = json.load(f)
    assert loaded_config == experiment_cfg_poison

    with open(experiment_dir / 'clean_result.json', 'r') as f:
        loaded_clean_result = json.load(f)
    assert loaded_clean_result == result['clean_result']

    with open(experiment_dir / 'perturbed_result.json', 'r') as f:
        loaded_perturbed_result = json.load(f)
    assert loaded_perturbed_result == result['perturbed_result']

def test_save_results_evasion_no_perturbed_csv(experiment_dir, experiment_cfg_evasion, result):
    experiment_logger = LogExperiment(experiment_dir, experiment_cfg_evasion, result, csv_save=True)
    experiment_logger.save_results()

    # No CSV for evasion attacks
    assert Path(experiment_dir / 'experiment_config.json').exists()
    assert Path(experiment_dir / 'clean_result.json').exists()
    assert Path(experiment_dir / 'clean_result.csv').exists()
    assert Path(experiment_dir / 'perturbed_result.json').exists()
    assert not Path(experiment_dir / 'perturbed_result.csv').exists()
    
    with open(experiment_dir / 'experiment_config.json', 'r') as f:
        loaded_config = json.load(f)
    assert loaded_config == experiment_cfg_evasion

    with open(experiment_dir / 'clean_result.json', 'r') as f:
        loaded_clean_result = json.load(f)
    assert loaded_clean_result == result['clean_result']

    with open(experiment_dir / 'perturbed_result.json', 'r') as f:
        loaded_perturbed_result = json.load(f)
    assert loaded_perturbed_result == result['perturbed_result']


def test_save_results_no_csv(experiment_dir, experiment_cfg_poison, result):
    experiment_logger = LogExperiment(experiment_dir, experiment_cfg_poison, result, csv_save=False)
    experiment_logger.save_results()

    # No CSV at all
    assert Path(experiment_dir / 'experiment_config.json').exists()
    assert Path(experiment_dir / 'clean_result.json').exists()
    assert not Path(experiment_dir / 'clean_result.csv').exists()
    assert Path(experiment_dir / 'perturbed_result.json').exists()
    assert not Path(experiment_dir / 'perturbed_result.csv').exists()

@patch('gnn_toolbox.experiment_handler.result_saver.open', side_effect=IOError)
def test_save_results_io_error(mock_open, experiment_dir, experiment_cfg_poison, result, caplog):
    experiment_logger = LogExperiment(experiment_dir, experiment_cfg_poison, result, csv_save=True)
    experiment_logger.save_results()

    assert "Failed to log experiment configuration" in caplog.text
    assert "Failed to log results" in caplog.text