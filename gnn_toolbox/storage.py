from tinydb import TinyDB, Query
from pathlib import Path
import json, hashlib
import torch
import logging
class Storage:
    def __init__(self, save_dir, experiment_name, model_name, dataset_name):
        self.save_dir = save_dir
        save_dir.mkdir(exist_ok=True)
        self.dbs = []
        self.dbs.append(TinyDB(save_dir / experiment_name / f'{model_name}_{dataset_name}.json'))
        
    def save_artifact(self, artifact, name, params):
        for db in self.dbs:
            db.insert({name: artifact, 'params': params})
        
    def load_artifact(self, name):
        pass
# f = Path('test.json')
# db = TinyDB(f)

# db.insert({'int': 1, 'char': 'a'})

class ArtifactManager:
    def __init__(self, cache_directory):
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(exist_ok=True)
    
    def folder_exists(self, folder_path):
        folder = Path(folder_path)
        return folder.exists() and folder.is_dir()    
    
    def hash_parameters(self, params):
        params_string = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_string.encode()).hexdigest()
        
    # def check_model(self, cfg):
    #     # * hardcoded 'attack', might be bad
    #     new_cfg = {key: value for key, value in original_dict.items() if key != 'attack'}
    #     self.model_exists(new_cfg)
        
    def save_model(self, model, params, result, is_unattacked_model):
        if is_unattacked_model:
            params = {key: value for key, value in params.items() if key != 'attack'}
            model_suffix = f"{params['model']['name']}_{params['dataset']['name']}.pt"
            result_file_name = "result.json"
        else:
            model_suffix = f"{params['attack']['name']}_{params['model']['name']}_{params['dataset']['name']}.pt"
            result_file_name = "attacked_result.json"

        hash_id = self.hash_parameters(params)
        model_dir = self.cache_directory / f"{hash_id}"
        model_path = model_dir / model_suffix
        result_path = model_dir / result_file_name
        params_path = model_dir / "params.json"

        model_dir.mkdir(exist_ok=True)

        try:
            torch.save(model.state_dict(), model_path)
            with params_path.open('w') as file:
                json.dump(params, file, indent=4)
            with result_path.open('w') as file:
                json.dump(result, file, indent=4)
        except Exception as e:
            logging.error(f"Failed to save model or results: {e}")
        else:
            logging.info('SAVED MODEL')

    def model_exists(self, experiment_params):
        """ Check if a model with the given parameters already exists. """
        params = {key: value for key, value in experiment_params.items() if key != 'attack'}
        hash_id = self.hash_parameters(params)
        params_dir = self.cache_directory / f"{hash_id}"
        if self.folder_exists(params_dir):
            params_path = params_dir / 'params.json'
            with params_path.open('r') as file:
                saved_params = json.load(file)
            if params == saved_params: # * i need deepdiff
                model_path = params_dir / f"{params['model']['name']}_{params['dataset']['name']}.pt"
                result_path = params_dir / 'result.json'
                print('FOUND MODEL')
                if result_path.exists():
                    with result_path.open('r') as file:
                        result = json.load(file)
                    print('FOUND RESULT')
                    return model_path, result
        return None, None
        # model_path = self.cache_directory / f"{hash_id}.pt"

    


    
    
        

# params = {'a': 1, 'b': 2}
# params2 = {'a': 1, 'b': 3}
# params3 = {'b': 2, 'a':1}
# params_string = json.dumps(params, sort_keys=True)
# print(hashlib.sha256(params_string.encode()).hexdigest())
# params_stringw = json.dumps(params2, sort_keys=True)
# print(hashlib.sha256(params_stringw.encode()).hexdigest())
# params_string3 = json.dumps(params3, sort_keys=True)
# print(hashlib.sha256(params_string3.encode()).hexdigest())