import yaml
import torch
import logging
from typing import List, Dict, Union, Optional, Literal
from pydantic import BaseModel, ConfigDict, model_validator, PositiveInt, PositiveFloat, NonNegativeInt, field_validator
import yaml.parser
from gnn_toolbox.registry import registry
from custom_components import *

PARAMS_TYPE = Union[List[int], int, List[float], float, List[str], str, List[bool], bool]


def load_and_validate_yaml(yaml_path: str):
    """Loads and validates the YAML file at the given path.

    Args:
        yaml_path (str): path of YAML file

    Raises:
        yaml.YAMLError: Raised when failed to parse YAML file
        FileExistsError: Raised when failed to find or load YAML file
        ValueError: Raised when validation error(s) encountered

    Returns:
        Dict: dictionary of validated YAML file
    """
    try:
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
    except (yaml.YAMLError) as e:
        raise yaml.YAMLError(f"Failed to parse YAML file at {yaml_path}.") from e
    except Exception as e:
        raise FileExistsError(f"Failed to find or load YAML file at the location: {yaml_path}.") from e
    try:
        config = Config(**yaml_data)
        logging.info(f"Given YAML file at {yaml_path} is valid, generating experiments.")
        dict_config = config.model_dump()
        return dict_config
    except Exception as e:
        for error in e.errors():
            if error['msg'].startswith('Input should be a valid dictionary or instance of'):
                continue
            logging.error(format_error_message(error))
            
        raise ValueError("Validation error(s) encountered. See logs for details.") from e

def check_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

class Model(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Union[str, List[str]]
    params: Optional[Dict[str, PARAMS_TYPE]] = {}
    @model_validator(mode='after')
    def name_must_be_registered(self):
        check_if_value_registered(self.name, 'model')
        return self

class Optimizer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Union[str, List[str]]
    params: Optional[Dict[str, PARAMS_TYPE]] = {}
    @model_validator(mode='after')
    def name_must_be_registered(self):
        check_if_value_registered(self.name, 'optimizer')
        return self

class Transform(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    params: Optional[Dict[str, PARAMS_TYPE]] = {}
    @model_validator(mode='after')
    def name_must_be_registered(self):
        check_if_value_registered(self.name, 'transform')
        return self

class Training(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_epochs: PositiveInt
    patience: PositiveInt
    @model_validator(mode='after')
    def validate_values(self):
        if self.max_epochs < self.patience:
            raise ValueError("Max epochs must be greater than patience.")
        return self
    
class Loss(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Union[str, List[str]]
    params: Optional[Dict[str, PARAMS_TYPE]] = {}
    @model_validator(mode='after')
    def name_must_be_registered(self):
        check_if_value_registered(self.name, 'loss')
        return self

class Dataset(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Union[str, List[str]]
    root: str = './datasets'
    make_undirected: Optional[bool] = False
    graph_index: Optional[PositiveInt] = 0
    transforms: Optional[Union[Transform, List[Transform]]]  = None
    train_ratio: Optional[PositiveFloat] = None
    val_ratio: Optional[PositiveFloat] = None
    test_ratio: Optional[PositiveFloat] = None
    params: Optional[Dict[str, PARAMS_TYPE]] = {}
    
    @field_validator('train_ratio', 'val_ratio', 'test_ratio')
    def check_ratio_range(cls, value):
        if value is not None and (value < 0 or value > 1):
            raise ValueError("Dataset split ratios must be between 0 and 1.")
        return value
    
    @model_validator(mode='after')
    def name_must_be_registered(self):
        check_if_value_registered(self.name, 'dataset')
        self.train_ratio, self.val_ratio, self.test_ratio = check_ratios(self.train_ratio, self.val_ratio, self.test_ratio)
        return self

def check_ratios(train_ratio, val_ratio, test_ratio):
    provided_ratios = [r for r in [train_ratio, val_ratio, test_ratio] if r is not None]
    if len(provided_ratios) == 1:
        raise ValueError("At least two of train_ratio, val_ratio, and test_ratio must be provided if any are given.")
    elif len(provided_ratios) == 2:
        total_provided = sum(provided_ratios)
        if total_provided > 1.0:
            raise ValueError(f"train_ratio, val_ratio, and test_ratio sum must be less or equal to 1.0 but the sum is {total_provided}.")
        leftover_ratio = 1.0 - total_provided
        if train_ratio is None:
            train_ratio = leftover_ratio
        elif val_ratio is None:
            val_ratio = leftover_ratio
        else:
            test_ratio = leftover_ratio
    elif len(provided_ratios) == 3:
        total_provided = sum(provided_ratios)
        if total_provided > 1.0:
            raise ValueError(f"train_ratio, val_ratio, and test_ratio sum must be less or equal to 1.0 but the sum is {total_provided}.")
    return train_ratio, val_ratio, test_ratio
        
    
def check_if_value_registered(value, key):
    if isinstance(value, List):
        for name in value:
            if name not in registry[key].keys():
                raise ValueError(f"Invalid model name '{name}'. Must be one of: {list(registry[key].keys())}")
    elif value not in registry[key].keys():
        raise ValueError(f"Invalid model name '{value}'. Must be one of: {list(registry[key].keys())}")

class Attack(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scope: Literal['local', 'global']
    name: Union[str, List[str]]
    type: Literal['evasion', 'poison']
    epsilon: Union[List[Union[PositiveInt, PositiveFloat]], PositiveInt, PositiveFloat]
    nodes: Optional[List[NonNegativeInt]] = None
    params: Optional[Dict[str, PARAMS_TYPE]] = {}
    @model_validator(mode='after')
    def validate_scope(self):
        if self.scope == 'local':
            if self.nodes is None:
                raise ValueError("For local scope, 'nodes' must be provided.")
            check_if_value_registered(self.name, 'local_attack')
        elif self.scope == 'global':
            check_if_value_registered(self.name, 'global_attack')
        return self


class ExperimentTemplate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str 
    seed: Union[List[int], int] = [0, 1]
    device: Optional[Literal['cpu', 'cuda']] = check_device()
    model: Union[Model, List[Model]]
    dataset: Union[Dataset, List[Dataset]]
    attack: Optional[Union[Attack, List[Attack]]]
    training: Union[Training, List[Training]]
    optimizer: Union[Optimizer, List[Optimizer]]
    loss: Union[Loss, List[Loss]]

class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')
    output_dir: str = './output'
    cache_dir: str = './cache'
    resume_output: Optional[bool] = False
    csv_save: Optional[bool] = True
    experiment_templates: List[ExperimentTemplate]

def format_error_message(error):
    loc = f'{error["loc"][0]} index {error["loc"][1]}, {error["loc"][2]}'
    return f"Validation error location: '{loc}'. \n Error type: {error['type']} \n Error message: {error['msg']} \n Input that caused error: '{error['input']}'."