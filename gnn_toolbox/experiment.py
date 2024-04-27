import os
import yaml
from itertools import product
from utils import DotDict
from pathlib import Path
CONFIG_VALUES = ['output_dir', 
                 'experiment_templates',
                 ]

class InputError(SystemExit):
    """Parent class for input errors that don't print a stack trace."""
    pass


class ConfigError(InputError):
    """Raised when the something is wrong in the config"""

    def __init__(self, message="The config file contains an error."):
        super().__init__(f"CONFIG ERROR: {message}")

def file_handler(file):
    with open(file, "r") as f:
        experiments_config = yaml.safe_load(f)
    
    for k in experiments_config.keys():
        if k not in CONFIG_VALUES:
            raise ConfigError(f"{k} is not a valid value in the `seml` config block.") 
    
    output_dir = experiments_config['output_dir']
    del experiments_config['output_dir']
    # experiments_template = experiments_config['experiments_template']
    
    # ! might be bad
    return output_dir, experiments_config['experiment_templates']
    
def flatten(dictionary: dict, parent_key: str = '', sep: str = '.'):
    """
    Flatten a nested dictionary, e.g. {'a':{'b': 2}, 'c': 3} becomes {'a.b':2, 'c':3}.
    From https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

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


def generate_experiments_from_yaml(yaml_file):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # # Join the directory path and your file name
    file_path = os.path.join(dir_path, yaml_file)
    with open(file_path, "r") as f:
        experiments_config = yaml.safe_load(f)

    def _generate_combinations(config_dict):
        parameter_dict = { key:value for key, value in config_dict.items() if isinstance(value, list)}
        for combination in product(*(parameter_dict.values())):
            yield dict(zip(parameter_dict.keys(), combination))

    output_dir = Path(experiments_config["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    all_experiments = []
    experiment_dirs = []
    for experiment_template in experiments_config["experiment_templates"]:
        experiment_template = flatten(experiment_template)
        combinations = list(_generate_combinations(experiment_template))
        experiments = [experiment_template.copy() for _ in range(len(combinations))]
        for id, (experiment, combination) in enumerate(zip(experiments, combinations)):
            for key, value in combination.items():
                experiment[key] = value
            experiment = DotDict(unflatten(experiment))
            experiment_dir = setup_directories(experiments_config['output_dir'],experiment['name'], id)
            all_experiments.append(experiment)
            experiment_dirs.append(experiment_dir)

    return all_experiments, experiment_dirs, output_dir

dir_path = os.path.dirname(os.path.realpath(__file__))

# def prepare_output_dir(base_dir, experiment):
#     base_path = Path(base_dir) / experiment
#     base_path.mkdir(parents=True, exist_ok=True)
#     return base_path

def setup_directories(base_dir, experiment_name, id):
    base_dir = Path(base_dir)
    experiment_dir = base_dir / experiment_name
    
    # existing_dirs = [d for d in base_dir.glob(experiment_name + '*') if d.is_dir()]
    # id = 0
    # if existing_dirs:
    #     # Extract suffixes and find the maximum number
    #     id = max([int(Path(existing_dir).name.split('_')[-1]) for existing_dir in existing_dirs]) + 1
    
    # Create a new directory with the next suffix
    new_experiment_dir = base_dir / f"{experiment_name}_{id}"
    new_experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return new_experiment_dir

def make_experiment_dir(experiment):
    pass    

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


# # Join the directory path and your file name
file_path = os.path.join(dir_path, 'good.yaml')
b = generate_experiments_from_yaml(file_path)
len(b)
i = b[0]
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

d['model']['params'].update({
    'sda':2
})
# print(d)

# next step is to use this dict to run experiment
# find how to connect defined modules to run experiment using sacred or yacs
# print(new_d)