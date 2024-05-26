from pathlib import Path
import json, hashlib
import torch
import logging

class ArtifactManager:
    def __init__(self, cache_directory):
        self.cache_directory = cache_directory
        self.cache_directory.mkdir(exist_ok=True)
    
    def folder_exists(self, folder_path):
        folder = Path(folder_path)
        return folder.exists() and folder.is_dir()    
    
    def hash_parameters(self, params):
        params_string = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_string.encode()).hexdigest()
                
    def save_model(self, model, params, result, is_unattacked_model):
        if is_unattacked_model:
            params = {key: value for key, value in params.items() if key != 'attack'}
            model_suffix = f"{params['model']['name']}_{params['dataset']['name']}.pt"
            result_file_name = "clean_result.json"
        else:
            model_suffix = f"{params['model']['name']}_{params['dataset']['name']}_{params['attack']['name']}.pt"
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
            logging.error(f"Failed to save model or results to {model_dir}: {e}")
        else:
            logging.info(f'Saved the model {model_suffix} to {model_dir} for caching')

    def model_exists(self, params, is_unattacked_model):
        """ Check if a model with the given parameters already exists. """
        if is_unattacked_model:
            params = {key: value for key, value in params.items() if key != 'attack'}
        hash_id = self.hash_parameters(params)
        params_dir = self.cache_directory / f"{hash_id}"
        if self.folder_exists(params_dir):
            if is_unattacked_model:
                model_name = f"{params['model']['name']}_{params['dataset']['name']}.pt"
                model_path = params_dir / model_name
                result_path = params_dir / 'clean_result.json'
                logging.info(f'Found the unattacked trained model at {model_path}')
                if result_path.exists():
                    with result_path.open('r') as file:
                        result = json.load(file)
                    logging.info(f'Found the {model_name} training result at {result_path}')
                else:
                    result = None
            else:
                attacked_model_name = f"{params['model']['name']}_{params['dataset']['name']}_{params['attack']['name']}.pt"
                model_path = params_dir / attacked_model_name
                result_path = params_dir / 'attacked_result.json'
                logging.info(f'Found the poisoned model at {model_path}')
                if result_path.exists():
                    with result_path.open('r') as file:
                        result = json.load(file)
                    logging.info(f'Found the poisoned {attacked_model_name} training result at {result_path}')
                else:
                    result = None
            return model_path, result
        return None, None

    
def hash_parameters(params):
    params_string = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_string.encode()).hexdigest()

a = {'a':{
    'wtf': 1,
    'c':{
        'd':{
            'f': 1,
        },
        'w': 1,
        'e': 2
    },
    }
}
    
a2 = {'a':{
    'c':{
        'e': 2,
        'w': 1,
        'd':{
            'f': 1
        }
    },
    'wtf': 1
    }
}    
a2_ =json.dumps(a2, sort_keys=True)
a_ = (json.dumps(a, sort_keys=True))
a_dict = json.loads(a_)
# print(a_dict['a'])
# if a_ == a2_:
#     print('yes')
# else:
#     print('no')
# print(hash_parameters(a))
# print(hash_parameters(a2))
# print(hash_parameters(a) == hash_parameters(a2))

# params = {'a': 1, 'b': 2}
# params2 = {'a': 1, 'b': 3}
# params3 = {'b': 2, 'a':1}
# params_string = json.dumps(params, sort_keys=True)
# print(hashlib.sha256(params_string.encode()).hexdigest())
# params_stringw = json.dumps(params2, sort_keys=True)
# print(hashlib.sha256(params_stringw.encode()).hexdigest())
# params_string3 = json.dumps(params3, sort_keys=True)
# print(hashlib.sha256(params_string3.encode()).hexdigest())