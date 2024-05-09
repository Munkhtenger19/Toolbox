import yaml
import torch
from datetime import datetime
# from cerberus import Validator
from pydantic import BaseModel, Field, PositiveInt, model_validator, ConfigDict
from typing import List, Union, Optional, Dict, Literal, Any, TypeVar, Generic, Iterator
def validate_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        experiments_config = yaml.safe_load(f)

def check_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

class Model(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    params: Optional[Dict[str, Any]] = {}

class Optimizer(BaseModel):
    name: str = 'adam'
    params: Optional[Dict[str, Any]] = {}

class Transform(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = {}

class Training(BaseModel):
    max_epochs: PositiveInt
    patience: PositiveInt

class Loss(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = {}

class Dataset(BaseModel):
    name: str
    root: str = './datasets'
    make_undirected: Optional[bool] = False
    graph_index: Optional[int] = 0
    transforms: Optional[Union[Transform, List[Transform]]]  = None

class Attack(BaseModel):
    scope: Literal['local', 'global']
    name: str
    type: Literal['evasion', 'poison']
    epsilon: Union[List[Union[float, int]], float, int]
    nodes: Optional[List[int]] = None
    min_node_degree: Optional[int] = None
    topk: Optional[int] =None
    
    @model_validator(mode='after')
    def validate_scope(self):
        scope = self.scope
        nodes = self.nodes
        min_node_degree = self.min_node_degree
        topk = self.topk
        epsilon = self.epsilon

        if scope == 'local':
            # if Nodes has some value or min_node_degree and topk are provided
            if nodes is None and (min_node_degree is None or topk is None):
                raise ValueError("For local scope, either 'nodes' or both 'min_node_degree' and 'topk' must be provided.")
        elif scope == 'global':
            if not epsilon:
                raise ValueError("For global scope, 'epsilon' must be provided.")
        else:
            raise ValueError("Invalid value for 'scope'. Must be either 'local' or 'global'.")
        
        return self

class ExperimentTemplate(BaseModel):
    # give current date and time
    name: str 
    # = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    seed: Union[List[int], int] = [0, 1]
    resume_output: Optional[bool] = False
    csv_save: bool = Field(default = True)
    device: Optional[Literal['cpu', 'cuda']] = check_device()
    model: Union[Model, List[Model]]
    dataset: Union[Dataset, List[Dataset]]
    attack: Union[Attack, List[Attack]]
    training: Union[Training, List[Training]]
    optimizer: Union[Optimizer, List[Optimizer]]
    loss: Union[Loss, List[Loss]]



class Config(BaseModel):
    output_dir: str
    experiment_templates: List[ExperimentTemplate]

def load_and_validate_yaml(yaml_path: str):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    try:
        config = Config(**yaml_data)
        print("YAML is valid.")
        return config
    except Exception as e:
        print('fuck')
        print(f"Validation Error: {e}")
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
# # # Join the directory path and your file name
file_path = os.path.join(dir_path, 'good_1.yaml')

a = load_and_validate_yaml(file_path)
print('a'   , a)
# a['model'][0]
schema = {
    'output'
    'name': {'type': 'string', 'required': True},
    'seed': {
        'anyof_type': ['list', 'string'],
        'schema': {'type': 'integer'},
        'required': False
    },
    'device': {'type': 'string', 'default': 'cpu'},
    'model': {
        'type': 'dict', 'required': True,
        'schema': {
            'name': {'type': 'string', 'required': True},
            'params': {'type': 'dict', 'required': False}
        }
    },
    # Other fields remain the same
}
