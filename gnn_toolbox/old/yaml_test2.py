import yaml
from itertools import product
import os
import json
import ast
def restore(flat):
    """
    Restore more complex data that Python's json can't handle (e.g. Numpy arrays).
    Copied from sacred.serializer for performance reasons.
    """
    return jsonpickle.decode(json.dumps(flat), keys=True)

def _convert_value(value):
    """
    Parse string as python literal if possible and fallback to string.
    Copied from sacred.arg_parser for performance reasons.
    """

    try:
        return restore(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        # use as string if nothing else worked
        return value

def convert_values(val):
    if isinstance(val, dict):
        for key, inner_val in val.items():
            val[key] = convert_values(inner_val)
    elif isinstance(val, list):
        for i, inner_val in enumerate(val):
            val[i] = convert_values(inner_val)
    elif isinstance(val, str):
        return _convert_value(val)
    # print('val', val)
    return val



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
    import collections

    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



dir_path = os.path.dirname(os.path.realpath(__file__))

# # Join the directory path and your file name
file_path = os.path.join(dir_path, 'good.yaml')

with open(file_path) as f:
    templates = yaml.safe_load(f)

a = convert_values(templates)
print(a)
print('templates', templates)
# qw = flatten(templates['experiments_template'][0])
# print(qw)
# convert_values(templates)



def handler(file):
    with open(file, "r") as f:
        experiments_config = yaml.safe_load(f)
    output_dir = experiments_config['output_dir']
    for key, values in dict1.items():
        if isinstance(values, list):
            pass

def generate_experiments_from_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        experiments_config = yaml.safe_load(f)

    def _generate_combinations(config_dict):
        parameter_lists = [value for key, value in config_dict.items() if isinstance(value, list)]
        for combination in product(*parameter_lists):
            yield dict(zip(config_dict.keys(), combination))

    def _overwrite_template_values(experiments_template, combinations, parameter_dict):
        def _update_nested_dict(d, keys, value):
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value

        for experiment, combination in zip(experiments_template, combinations):
            for i, (key_path, _) in enumerate(parameter_dict.items()):
                keys = key_path.split(".")  # Split key path into nested keys
                _update_nested_dict(experiment, keys, combination[i])
        return experiments_template

    all_experiments = []
    for experiment_template in experiments_config["experiments_template"]:
        parameter_dict = {}
        queue = [("", experiment_template, parameter_dict)]  # Start with an empty prefix
        while queue:
            prefix, current_dict, param_dict = queue.pop(0)
            for key, value in current_dict.items():
                new_prefix = f"{prefix}.{key}" if prefix else key  # Build nested key path
                if isinstance(value, dict):
                    queue.append((new_prefix, value, param_dict))
                elif isinstance(value, list):
                    param_dict[new_prefix] = value


        combinations = list(_generate_combinations(parameter_dict))
        experiments = _overwrite_template_values([experiment_template.copy() for _ in range(len(combinations))], combinations, parameter_dict)
        all_experiments.extend(experiments)

    return all_experiments

# experiments = generate_experiments_from_yaml("good.yaml")

# for experiment in experiments:
#     print(experiment)