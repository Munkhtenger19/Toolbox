from pathlib import Path
from itertools import product
import shutil

KEEP_AS_LIST = ['attack.nodes', 'dataset.transforms']

    
def flatten(dictionary: dict, parent_key: str = '', sep: str = '.'):
    """
    Flatten a nested dictionary, e.g. {'a':{'b': 2}, 'c': 3} becomes {'a.b':2, 'c':3}.
    Adapted from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Arguments
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
        elif value is not None:
            items.append((new_key, value))
    return dict(items)

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

def _generate_combinations(config_dict):
    parameter_dict = { key:value for key, value in config_dict.items() if isinstance(value, list) and key not in KEEP_AS_LIST}
    for combination in product(*(parameter_dict.values())):
        yield dict(zip(parameter_dict.keys(), combination))

def folder_exists(folder_path):
    folder = Path(folder_path)
    return folder.exists() and folder.is_dir()  

def generate_experiments_from_yaml(experiments_config: dict):
    """Generate experiments from a given experiment configuration dictionary.

    Args:
        experiments_config (dict): Experiment configuration dictionary

    Returns:
        dict: Dictionary of experiment directories and configurations
    """
    output_dir = Path(experiments_config["output_dir"])
    cache_dir = Path(experiments_config["cache_dir"])
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
            all_experiments[experiment_dir] = experiment
            id+=1
            
    return all_experiments, cache_dir

def setup_directories(base_dir: str, experiment_name: str, id: int, resume_output: bool):
    """Create a new experiment directory.

    Args:
        base_dir (str): main directory for storing experiments subdirectories
        experiment_name (str): name of the experiment
        id (int): id of the experiment
        resume_output (bool): whether to resume output

    Returns:
        Path: Path to the new experiment directory
    """
    base_dir = Path(base_dir)
    
    if resume_output:
        existing_dirs = [d for d in base_dir.glob(experiment_name + '*') if d.is_dir()]
        if existing_dirs:
            id = max([int(Path(existing_dir).name.split('_')[-1]) for existing_dir in existing_dirs]) + 1
    
    new_experiment_dir = base_dir / f"{experiment_name}_{id}"
    new_experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return new_experiment_dir