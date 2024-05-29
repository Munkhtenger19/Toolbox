# test_exp_gen.py
import pytest
import os
import shutil
from pathlib import Path
from gnn_toolbox.experiment_handler.exp_gen import (
    flatten,
    _generate_combinations,
    folder_exists,
    generate_experiments_from_yaml,
    setup_directories,
    unflatten,
)

@pytest.fixture
def test_output_dir():
    directory = Path('test_output')
    directory.mkdir(exist_ok=True)
    yield directory
    shutil.rmtree(directory)


# Experiment config with models, datasets, attacks, training, and optimizers each with dictionaries. Inside those dictionaries, some of them have lists.
@pytest.fixture
def experiment_config_nested(test_output_dir):
    return {
        'output_dir': 'test_output',
        'cache_dir': 'test_cache',
        'experiment_templates': [
            {
                'name': 'TestExperiment',
                'seed': ['0', '1'],
                'model': {
                    'name': ['GCN', 'GAT'],
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
                    'type': 'evasion',
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
        ]
    }

# Experiment config with multiple models, datasets, attacks, training, and optimizers with lists in multiple levels deep. 
# For example, the 'model' key defines a list and that list also contains lists.
@pytest.fixture
def experiment_config_deep_nested(test_output_dir):
    return {
        'output_dir': 'test_output',
        'cache_dir': 'test_cache1',
        'experiment_templates': [
            {
                'name': 'Local attack_3',
                'seed': ['0', '1'],
                'device': 'cuda',
                'model': [
                    {
                        'name': ['GCN1', 'GCN2'],
                        'params': {
                            'hidden_channels': '64'
                        }
                    },
                    {
                        'name': ['GAT1', 'GAT2'],
                        'params': {
                            'hidden_channels': '32'
                        }
                    }
                ],
                'dataset': [
                    {
                        'name': ['Cora', 'CiteSeer', 'PubMed'], 
                        'root': './datasets',
                        'make_undirected': 'true',
                        'train_ratio': '0.1',
                        'test_ratio': '0.4'
                    },
                    {
                        'name': ['Cora', 'CiteSeer', 'PubMed'], 
                        'root': './datasets',
                        'make_undirected': 'false',
                        'train_ratio': '0.2',
                        'test_ratio': '0.3'
                    }
                ],
                'attack': [
                    {
                        'scope': 'local',
                        'type': 'poison',
                        'name': 'LocalDICE',
                        'epsilon': ['0.1', '0.2'],
                        'nodes': ['1', '2', '3', '4', '5']
                    },
                    {
                        'scope': 'global',
                        'type': 'evasion',
                        'name': 'GlobalDICE',
                        'epsilon': ['0.1', '0.2'],
                    }
                ],
                'training': [
                    {
                        'max_epochs': '100',
                        'patience': '80'
                    },
                    {
                        'max_epochs': '200',
                        'patience': '160'
                    }
                ],
                'optimizer': [
                    {
                        'name': ['Adam', 'sgd'],
                        'params': {
                            'lr': '0.01',
                            'weight_decay': '0.0005'
                        }
                    },
                    {
                        'name': ['AdamW', 'PAdam'],
                        'params': {
                            'lr': '0.002',
                            'weight_decay': '0.00004'
                        }
                    }
                ],
                'loss': [
                    {
                        'name': ['CE', 'MSE'],
                    }
                ]
            }
        ]
    }

def test_flatten():
    nested_dict = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': {'f': {'g': 4}}}
    flattened_dict = flatten(nested_dict)
    assert flattened_dict == {'a': 1, 'b.c': 2, 'b.d': 3, 'e.f.g': 4}

def test_unflatten():
    flattened_dict = {'a': 1, 'b.c': 2, 'b.d': 3, 'e.f.g': 4}
    nested_dict = unflatten(flattened_dict)
    assert nested_dict == {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': {'f': {'g': 4}}}

def test_generate_combinations():
    config_dict = {'a': [1, 2], 'b.c': [3, 4], 'd': 5}
    combinations = list(_generate_combinations(config_dict))
    assert len(combinations) == 4
    assert {'a': 1, 'b.c': 3} in combinations 
    assert {'a': 1, 'b.c': 4} in combinations
    assert {'a': 2, 'b.c': 3} in combinations
    assert {'a': 2, 'b.c': 4} in combinations

def test_folder_exists(test_output_dir):
    assert folder_exists(test_output_dir)
    assert not folder_exists('non_existing_folder')

def test_generate_experiments_from_yaml_nested_config(experiment_config_nested, experiment_config_deep_nested):
    experiments, cache_dir = generate_experiments_from_yaml(experiment_config_nested)
    assert len(experiments) == 48  # 2 seeds * 2 models * 3 datasets * 2 epsilons * 2 optimizers
    assert Path(cache_dir) == Path('test_cache')
    
    for experiment_dir, experiment in experiments.items():
        assert experiment_dir.is_dir()
        assert 'name' in experiment
        assert 'seed' in experiment
        assert 'model' in experiment
        assert 'dataset' in experiment
        assert 'attack' in experiment
        assert 'training' in experiment
        assert 'optimizer' in experiment
        assert 'loss' in experiment
        
def test_generate_experiments_from_yaml_deep_nested_config(experiment_config_deep_nested):
    experiments, cache_dir = generate_experiments_from_yaml(experiment_config_deep_nested)
    assert len(experiments) == 3072  # 2 seeds * 4 models * 6 datasets * 4 attacks * 2 trainings * 4 optimizers * 2 epsilons
    assert Path(cache_dir) == Path('test_cache1')
    
    for experiment_dir, experiment in experiments.items():
        assert experiment_dir.is_dir()
        assert 'name' in experiment
        assert 'seed' in experiment
        assert 'model' in experiment
        assert 'dataset' in experiment
        assert 'attack' in experiment
        assert 'training' in experiment
        assert 'optimizer' in experiment
        assert 'loss' in experiment

def test_generate_experiments_from_yaml_resuming(test_output_dir, experiment_config_nested):
    # Create a partial experiment output directory
    experiment_config_nested['resume_output'] = True
    experiments, _ = generate_experiments_from_yaml(experiment_config_nested)
    assert len(experiments) == 48
    
    dir_ids = [int(os.path.basename(path).split('_')[1]) for path in experiments.keys()]
    assert sorted(dir_ids) == list(range(1, 49))
    
    experiments, _ = generate_experiments_from_yaml(experiment_config_nested)
    assert len(experiments) == 48
    
    dir_ids = [int(os.path.basename(path).split('_')[1]) for path in experiments.keys()]
    assert sorted(dir_ids) == list(range(49, 97))

@pytest.mark.parametrize("resume_output", [True, False])
def test_setup_directories(test_output_dir, resume_output):
    experiment_name = 'TestExperiment'
    id = 1
    experiment_dir = setup_directories(test_output_dir, experiment_name, id, resume_output=resume_output)
    experiment_dir2 = setup_directories(test_output_dir, experiment_name, id, resume_output=resume_output)
    assert (experiment_dir).is_dir()
    assert (experiment_dir2).is_dir()
    
    if resume_output:
        assert test_output_dir / 'TestExperiment_2' == experiment_dir2
    else:
        assert test_output_dir / 'TestExperiment_1' == experiment_dir2

    
