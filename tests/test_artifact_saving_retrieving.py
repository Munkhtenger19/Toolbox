from pathlib import Path
import pytest
import os
import shutil
from unittest.mock import patch
from gnn_toolbox.experiment_handler.artifact_manager import ArtifactManager
import json
import torch

@pytest.fixture
def artifact_manager(tmp_path):
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    yield ArtifactManager(cache_dir)
    shutil.rmtree(cache_dir)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)
    
def test_folder_exists(artifact_manager):
    assert artifact_manager.folder_exists(artifact_manager.cache_directory)
    assert not artifact_manager.folder_exists('non_existing_folder')

def test_hash_parameters(artifact_manager):
    params1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
    params2 = {'b': {'d': 3, 'c': 2}, 'a': 1}  # Same as params1 but with different key order
    assert artifact_manager.hash_parameters(params1) == artifact_manager.hash_parameters(params2)

def test_save_model_unattacked(artifact_manager):
    model = Model()

    params = {'model': {'name': 'GCN'}, 'dataset': {'name': 'Cora'}, 'attack': {'name': 'DICE'}}
    result = {'accuracy': 0.85}

    # Test saving unattacked model
    artifact_manager.save_model(model, params, result, is_unattacked_model=True)

    params_with_no_attack = {key: value for key, value in params.items() if key != 'attack'}
    hash_id = artifact_manager.hash_parameters(params_with_no_attack)

    assert Path(artifact_manager.cache_directory / f"{hash_id}" / 'clean_result.json').exists()
    assert Path(artifact_manager.cache_directory / f"{hash_id}" / 'GCN_Cora.pt').exists()
    assert Path(artifact_manager.cache_directory / f"{hash_id}" / 'params.json').exists()
    with open(artifact_manager.cache_directory / f"{hash_id}" / 'params.json', 'r') as f:
        loaded_config = json.load(f)
    assert params_with_no_attack == loaded_config
    
    with open(artifact_manager.cache_directory / f"{hash_id}" / 'clean_result.json', 'r') as f:
        loaded_result = json.load(f)
    assert result == loaded_result

def test_save_model_attacked(artifact_manager):
    model = Model()

    params = {'model': {'name': 'GCN'}, 'dataset': {'name': 'Cora'}, 'attack': {'name': 'DICE'}}
    result = {'accuracy': 0.85}

    # Test saving unattacked model
    artifact_manager.save_model(model, params, result, is_unattacked_model=False)

    hash_id = artifact_manager.hash_parameters(params)
    
    assert Path(artifact_manager.cache_directory / f"{hash_id}" / 'attacked_result.json').exists()
    assert Path(artifact_manager.cache_directory / f"{hash_id}" / 'GCN_Cora_DICE.pt').exists()
    assert Path(artifact_manager.cache_directory / f"{hash_id}" / 'params.json').exists()
    
    with open(artifact_manager.cache_directory / f"{hash_id}" / 'params.json', 'r') as f:
        loaded_config = json.load(f)
    assert params == loaded_config
    
    with open(artifact_manager.cache_directory / f"{hash_id}" / 'attacked_result.json', 'r') as f:
        loaded_result = json.load(f)
    assert result == loaded_result

@patch('gnn_toolbox.experiment_handler.artifact_manager.torch.save')
def test_save_model_file_already_exists(mock_torch_save, artifact_manager):
    model = Model()
    params = {'model': {'name': 'GCN'}, 'dataset': {'name': 'Cora'}, 'attack': {'name': 'DICE'}}
    result = {'accuracy': 0.85}

    # Save the model once
    artifact_manager.save_model(model, params, result, is_unattacked_model=True)

    # Try saving again (should overwrite or raise an error)
    mock_torch_save.reset_mock()  # Reset the mock to check for new calls
    artifact_manager.save_model(model, params, result, is_unattacked_model=True)
    mock_torch_save.assert_called_once()  # Check save is called again (overwriting)

def test_model_exists(artifact_manager):
    model = Model()
    params = {'model': {'name': 'GCN'}, 'dataset': {'name': 'Cora'}, 'attack': {'name': 'DICE'}}
    result = {'accuracy': 0.85}

    # Save the models
    artifact_manager.save_model(model, params, result, is_unattacked_model=True)
    artifact_manager.save_model(model, params, result, is_unattacked_model=False)

    # Test if unattacked model exists
    model_path, loaded_result = artifact_manager.model_exists(params, is_unattacked_model=True)
    assert os.path.exists(model_path)
    assert loaded_result == result

    # Test if attacked model exists
    model_path, loaded_result = artifact_manager.model_exists(params, is_unattacked_model=False)
    assert os.path.exists(model_path)
    assert loaded_result == result

    # Test with non-existing model
    params['model']['name'] = 'NonExistingModel'
    model_path, loaded_result = artifact_manager.model_exists(params, is_unattacked_model=True)
    assert model_path is None
    assert loaded_result is None

def test_model_exists_non_existing_directory(artifact_manager):
    params = {'model': {'name': 'GCN'}, 'dataset': {'name': 'Cora'}, 'attack': {'name': 'DICE'}}

    model_path, loaded_result = artifact_manager.model_exists(params, is_unattacked_model=True)
    assert model_path is None
    assert loaded_result is None

def test_model_exists_partial_file_existence(artifact_manager):
    model = Model()
    params = {'model': {'name': 'GCN'}, 'dataset': {'name': 'Cora'}, 'attack': {'name': 'DICE'}}
    result = {'accuracy': 0.85}
    params_with_no_attack = {key: value for key, value in params.items() if key != 'attack'}
    # Save the model
    artifact_manager.save_model(model, params, result, is_unattacked_model=True)

    # Remove the result file
    hash_id = artifact_manager.hash_parameters(params_with_no_attack)
    result_path = os.path.join(artifact_manager.cache_directory, hash_id, 'clean_result.json')
    os.remove(result_path)

    model_path, loaded_result = artifact_manager.model_exists(params, is_unattacked_model=True)
    assert os.path.exists(model_path)
    assert loaded_result is None

def test_artifact_manager_empty_cache_directory(artifact_manager):
    # No issues are expected with empty cache directory
    pass  # This test simply ensures the constructor of ArtifactMananger class doesn't crash