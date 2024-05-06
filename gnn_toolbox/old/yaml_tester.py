import yaml
import os
from yacs.config import CfgNode as CN
cfg = CN(new_allowed=True)
# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))

# # Join the directory path and your file name
file_path = os.path.join(dir_path, 'good.yaml')

# with open(file_path, 'r') as file:
#     a = yaml.safe_load(file)
# print(a['experiments'][2])
from itertools import product
with open(file_path) as f:
    templates = yaml.safe_load(f)
    
    
    # one level deep BAD
# def generate_experiments(templates):
#     experiments = []
#     for template in templates:
#         param_grid = product( *[param_values for param_values in template.values() if isinstance(param_values, list)])
#         # print('hey', list(param_grid))
#         for params in param_grid:
#             print(params)
#             experiment = template.copy()  
#             for i, param_name in enumerate(template.keys()):
#                 if isinstance(template[param_name], list):
#                     experiment[param_name] = params[i]
#             experiments.append(experiment)
#     return experiments

a = {}
def gen_exp(exp_dict):
    for key, values in exp_dict.items():
        if(isinstance(values, dict)):
            gen_exp(values)
        elif(isinstance(values, list)):
            a[key] = values
            # copied = exp_dict.copy()
        else:
            print('value', values)


gen_exp(templates['experiments_template'][0])

combination = product(*[param_values for param_values in a.values()])

# odo gen hiitsen bolhor substitute hiine
def gen_exp2(exp_dict):
    for each in combination:
        copied = exp_dict.copy()
        cntr=0
        for key, value in a.items():
            for key2,value2 in exp_dict.items():
                if key == key2:
                    copied[key] = each[cntr]
                    cntr += 1
                    print('key, key2', key, key2)
            
gen_exp2(templates['experiments_template'][0])

def overwrite_template_values(experiments_template, combinations, parameter_dict):
    for experiment, combination in zip(experiments_template, combinations):
        for i, (key, value) in enumerate(parameter_dict.items()):
            if isinstance(value, list):
                # Overwrite the list value with the corresponding element from the tuple
                experiment[key] = combination[i]
    return experiments_template

# updated_experiments_template = overwrite_template_values(templates['experiments_template'][0], combination, a)
# for e in updated_experiments_template:
#     print(e)
# print(len(list(combination)))
   
# print(next(gen_exp(templates['experiments_template'][0])))
# print(next(gen_exp(templates['experiments_template'][0])))
# print(next(gen_exp(templates['experiments_template'][0])))
# gen_exp(templates['experiments_template'][0])
    
def generate_experiments(experiments_config):
    def _generate_combinations(config_dict):
        parameter_lists = [value for key, value in config_dict.items() if isinstance(value, list)]
        for combination in product(*parameter_lists):
            yield dict(zip(config_dict.keys(), combination))

    for experiment in experiments_config["experiments_template"]:
        # Recursively generate combinations for nested dictionaries
        queue = [experiment]
        while queue:
            current_dict = queue.pop(0)
            for key, value in list(current_dict.items()):  # Iterate over a copy of items
                if isinstance(value, dict):
                    queue.append(value)  # Process nested dictionaries later
                elif isinstance(value, list):
                    combinations = _generate_combinations({key: value})
                    for combination in combinations:
                        updated_dict = current_dict.copy()  # Create a copy to update
                        updated_dict.update(combination)
                        queue.append(updated_dict)  # Add updated dict back to the queue
        yield experiment.copy()  # Yield a copy to avoid modification

# def generate_experiment_combinations(config):
#     keys, values = zip(*config.items())
#     for combination in itertools.product(*values):
#         yield dict(zip(keys, combination))

# experiments = generate_experiments(templates)

# for experiment in experiments:
#     print(experiment)
# import itertools
# def generate_experiment_combinations(experiments_templates):
#   """
#   Generates experiment configurations by creating Cartesian products of parameters.

#   Args:
#     experiments_templates (list): List of experiment template dictionaries.

#   Returns:
#     list: A list of experiment configuration dictionaries.
#   """
#   all_combinations = []
#   for template in experiments_templates:
#     parameter_names = list(template.keys())
#     parameter_values = [template[name] for name in parameter_names]
#     combinations = list(itertools.product(*parameter_values))
#     all_combinations.extend([
#         {name: value for name, value in zip(parameter_names, combination)}
#         for combination in combinations
#     ])
#   return all_combinations

# # Example usage
# experiments = generate_experiment_combinations(templates['experiments_template'])
# for i, experiment in enumerate(experiments):
#   print(f"Experiment {i + 1}:")
#   print(experiment)