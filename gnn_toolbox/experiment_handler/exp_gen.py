import os
import yaml
import json
from pathlib import Path
from itertools import product
import shutil
import pprint


KEEP_AS_LIST =['attack.nodes', 'dataset.transforms']
CONFIG_VALUES = ['output_dir', 
                 'experiment_templates',
                 ]

# class ConfigManager:
#     _global_config = {}

#     @staticmethod
#     def update_config(new_config):
#         ConfigManager._global_config = new_config

#     @staticmethod
#     def get_config():
#         return ConfigManager._global_config

class InputError(SystemExit):
    """Parent class for input errors that don't print a stack trace."""
    pass


class ConfigError(InputError):
    """Raised when the something is wrong in the config"""

    def __init__(self, message="The config file contains an error."):
        super().__init__(f"CONFIG ERROR: {message}")
    
def flatten(dictionary: dict, parent_key: str = '', sep: str = '.'):
    """
    Flatten a nested dictionary, e.g. {'a':{'b': 2}, 'c': 3} becomes {'a.b':2, 'c':3}.
    Adapted from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Parameters
    ----------
    dictionary: dict to be flattened
    parent_key: string to prepend the key with
    sep: level separator

    Returns
    -------
    flattened dictionary.
    """
    from collections.abc import MutableMapping

    items = []
    for key, value in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def _generate_combinations(config_dict):
    # formatted_dict = { key:value[0] for key, value in config_dict.items() if isinstance(value, list) and len(value) == 1}
    parameter_dict = { key:value for key, value in config_dict.items() if isinstance(value, list) and key not in KEEP_AS_LIST}
    for combination in product(*(parameter_dict.values())):
        yield dict(zip(parameter_dict.keys(), combination))

def folder_exists(folder_path):
    folder = Path(folder_path)
    return folder.exists() and folder.is_dir()  

def generate_experiments_from_yaml(experiments_config):
    output_dir = Path(experiments_config["output_dir"])
    resume_output = experiments_config.get("resume_output", False)
    
    if (not resume_output) and folder_exists(output_dir):
        shutil.rmtree(output_dir)
        
    output_dir.mkdir(exist_ok=True)
    
    first_depth_exp_gen=[]
    for experiment_template in experiments_config["experiment_templates"]:
        experiment_template = flatten(experiment_template)
        combinations = list(_generate_combinations(experiment_template))
        experiments = [experiment_template.copy() for _ in range(len(combinations))]
        
        for experiment, combination in zip(experiments, combinations):
            for key, value in combination.items():
                experiment[key] = value
            experiment = unflatten(experiment)
            first_depth_exp_gen.append(experiment)
            
    all_experiments={}
    id=1
    for experiment in first_depth_exp_gen:
        experiment_template = flatten(experiment)
        combinations = list(_generate_combinations(experiment_template))
        experiments = [experiment_template.copy() for _ in range(len(combinations))]
        
        for experiment, combination in zip(experiments, combinations):
            for key, value in combination.items():
                experiment[key] = value
            experiment = unflatten(experiment)
            experiment_dir = setup_directories(experiments_config['output_dir'], experiment['name'], id, resume_output)
            # test2.append(experiment)
            # exp_test.append(experiment_dir)
            all_experiments[experiment_dir] = experiment
            id+=1
            
    return all_experiments

def setup_directories(base_dir, experiment_name, id, resume_output):
    base_dir = Path(base_dir)
    
    if resume_output:
        existing_dirs = [d for d in base_dir.glob(experiment_name + '*') if d.is_dir()]
        if existing_dirs:
            id = max([int(Path(existing_dir).name.split('_')[-1]) for existing_dir in existing_dirs]) + 1
    
    new_experiment_dir = base_dir / f"{experiment_name}_{id}"
    new_experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return new_experiment_dir
   
def unflatten(dictionary):
    """
    Unflatten a dictionary
    https://stackoverflow.com/questions/6037503/python-unflatten-dict
    """
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict

def tensorboard_logger():
    pass

# dir_path = os.path.dirname(os.path.realpath(__file__))
# # # # Join the directory path and your file name
# file_path = os.path.join(dir_path, 'good_1.yaml')
# b = generate_experiments_from_yaml(file_path)
# print(len(b))
# # print(i)
# def foo(lr):
#     print(lr)
# graph_index = i['dataset'].pop('graph_index', None)

# foo( **getattr(i['optimizer']['pz'], {'lr': 0.01}))

# print(i)
# print('graph_index', graph_index)
# for transform in i.dataset.transforms:
#     # desc = getattr(transform, 'name')
#     desc = transform.get('name')
    
#     print(desc)
# print(desc)
# print
# print(i)
# s = i.attack.node
# q = i.attack.params
# print(s, q)

# print()
# print(i.model.params)
# i['model']['params'].update({
#     'in_channels': 2,
#     'out_channels': 1,
# })
# print(i.model.params)
# i.model.params.in_channels= 2
# print(type(i['training']['lr']))
# print(i.model.params)




d = {
    'model': {
        'name': 'GCN',
        'params': {
            'in_channels': 1,
            'out_channels': 1,
            'hidden_channels': [32, 32],
            'num_layers': 2,
            'dropout': 0.5,
        }
    },
    'dataset': {
        'name': 'Cora',
        'params': {
            'root': 'data',
            'name': 'Cora',
        }
    },
    'trainer': {
        'device': 'cuda',
        'epochs': 1000,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'patience': 20,
        'display_step': 1,
    }

}

# q = (d['trainer']['wtf'])
# print(q)
# print(d)

# next step is to use this dict to run experiment
# find how to connect defined modules to run experiment using sacred or yacs
# print(new_d)