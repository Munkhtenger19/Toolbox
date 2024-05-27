import pytest
import os
import yaml
from gnn_toolbox.experiment_handler.config_validator import (
    load_and_validate_yaml,
    Config,
    ExperimentTemplate,
    Model,
    Dataset,
    Attack,
    Training,
    Optimizer,
    Loss,
    Transform,
    check_ratios,
)

############################ Fixtures ############################

@pytest.fixture
def valid_config_path():
    return os.path.join(os.path.dirname(__file__), 'test_configs', 'valid_config.yaml')

@pytest.fixture
def invalid_config_path():
    return os.path.join(os.path.dirname(__file__), 'test_configs', 'invalid_config.yaml')

@pytest.fixture(scope='module')
def mock_registry(monkeypatch):
    mock_registry_data = {
        'model': {'GCN': None, 'GCN2': None},
        'optimizer': {'Adam': None, 'SGD': None},
        'transform': {'NormalizeFeatures': None},
        'loss': {'CE': None, 'CW': None},
        'dataset': {'Cora': None, 'Citeseer': None},
        'global_attack': {'DICE': None, 'PRBCD': None},
        'local_attack': {'LocalDICE': None, 'LocalPRBCD': None},
    }
    monkeypatch.setattr('gnn_toolbox.registration_handler.registry.registry', mock_registry_data) # Mock the registry


##################### Experiment configuration file validation tests #####################

def test_load_and_validate_yaml_valid(valid_config_path):
    config = load_and_validate_yaml(valid_config_path)
    assert isinstance(config, dict)
    assert 'output_dir' in config
    assert 'experiment_templates' in config

def test_load_and_validate_yaml_invalid(invalid_config_path, caplog):
    with pytest.raises(ValueError) as excinfo:
        load_and_validate_yaml(invalid_config_path)
    assert "Validation error(s) encountered" in str(excinfo.value)
    assert len(caplog.records) > 0
    for record in caplog.records:
        assert record.levelname == "ERROR"

def test_load_and_validate_yaml_file_not_found(tmp_path):
    config_path = tmp_path / "nonexistent_config.yaml"
    with pytest.raises(FileExistsError) as excinfo:
        load_and_validate_yaml(str(config_path))
    assert "Failed to find or load YAML file" in str(excinfo.value)


##################### Validation classes tests #####################

def test_config_model():
    # Valid model configuration
    valid_model = Model(name='GCN', params={'hidden_channels': 64, 'dropout': 0.5})     
    assert valid_model.name == 'GCN'
    assert valid_model.params == {'hidden_channels': 64, 'dropout': 0.5}

    # Invalid
    with pytest.raises(ValueError) as excinfo:
        Model(name='InvalidModelName')
    assert "Invalid model name" in str(excinfo.value)

def test_config_dataset():
    # Valid dataset configuration
    valid_dataset = Dataset(name='Cora', root='./datasets')
    assert valid_dataset.name == 'Cora'
    assert valid_dataset.root == './datasets'

    # Invalid
    with pytest.raises(ValueError) as excinfo:
        Dataset(name='InvalidDatasetName')
    assert "Invalid model name" in str(excinfo.value)

@pytest.mark.parametrize("scope, attack_name, attack_type, epsilon, nodes, min_node_degree, nodes_topk", [
    ('global', 'DICE', 'evasion', 0.1, None, None, None),  # Valid global attack
    ('local', 'LocalDICE', 'evasion', 0.1, [1, 2, 3], None, None),  # Valid local attack with nodes
    ('local', 'LocalPRBCD', 'poison', 0.2, None, 2, 10),  # Valid local attack with min_node_degree and nodes_topk
])
def test_config_attack_valid(scope, attack_name, attack_type, epsilon, nodes, min_node_degree, nodes_topk):
    attack = Attack(scope=scope, type=attack_type, name=attack_name, epsilon=epsilon, nodes=nodes,
                    min_node_degree=min_node_degree, nodes_topk=nodes_topk)
    assert attack.scope == scope
    assert attack.name == attack_name

@pytest.mark.parametrize("scope, attack_name, attack_type, epsilon, nodes, min_node_degree, nodes_topk, error_msg", [
    ('invalid_scope', 'DICE', 'evasion', 0.1, None, None, None, "Input should be 'local' or 'global'"),  # Invalid attack scope
    ('global', 'InvalidAttackName', 'evasion', 0.1, None, None, None, "Invalid model name"),  # Invalid attack name
    ('local', 'LocalDICE', 'evasion', 0.1, None, None, None, "For local scope, either 'nodes' or both 'min_node_degree' and 'nodes_topk' must be provided."),  # Missing nodes for local attack
    ('local', 'LocalPRBCD', 'poison', 0.2, None, 2, None, "For local scope, either 'nodes' or both 'min_node_degree' and 'nodes_topk' must be provided."),  # Missing nodes_topk for local attack
    ('local', 'LocalPRBCD', 'poison', 0.2, None, None, 10, "For local scope, either 'nodes' or both 'min_node_degree' and 'nodes_topk' must be provided."),  # Missing min_node_degree for local attack
])
def test_config_attack_invalid(scope, attack_name, attack_type, epsilon, nodes, min_node_degree, nodes_topk, error_msg):
    with pytest.raises(ValueError) as excinfo:
        Attack(scope=scope, type=attack_type, name=attack_name, epsilon=epsilon, nodes=nodes,
               min_node_degree=min_node_degree, nodes_topk=nodes_topk)
    assert error_msg in str(excinfo.value)

@pytest.mark.parametrize("max_epochs, patience", [
    (200, 20),  # Valid training configuration
])
def test_config_training_valid(max_epochs, patience):
    training = Training(max_epochs=max_epochs, patience=patience)
    assert training.max_epochs == max_epochs
    assert training.patience == patience

@pytest.mark.parametrize("max_epochs, patience, error_msg", [
    (100, 150, "Max epochs must be greater than patience."),  # Invalid one (patience greater than max_epochs)
])
def test_config_training_invalid(max_epochs, patience, error_msg):
    with pytest.raises(ValueError) as excinfo:
        Training(max_epochs=max_epochs, patience=patience)
    assert error_msg in str(excinfo.value)

def test_config_optimizer():
    # Valid optimizer configuration
    valid_optimizer = Optimizer(name='Adam', params={'lr': 0.01})
    assert valid_optimizer.name == 'Adam'
    assert valid_optimizer.params == {'lr': 0.01}

def test_config_loss():
    # Valid loss configuration
    valid_loss = Loss(name='CE')
    assert valid_loss.name == 'CE'

def test_config_transform():
    # Valid transform configuration
    valid_transform = Transform(name='NormalizeFeatures')
    assert valid_transform.name == 'NormalizeFeatures'

def test_config_experiment_template():
    # Valid experiment template configuration
    valid_template = ExperimentTemplate(
        name='TestExperiment',
        seed=42,
        model=Model(name='GCN'),
        dataset=Dataset(name='Cora'),
        attack=Attack(scope='global', type='evasion', name='DICE', epsilon=0.1),
        training=Training(max_epochs=200, patience=20),
        optimizer=Optimizer(name='Adam'),
        loss=Loss(name='CE')
    )
    assert valid_template.name == 'TestExperiment'

def test_config_full():
    # Valid full configuration
    valid_config = Config(
        output_dir='./output',
        cache_dir='./cache',
        experiment_templates=[
            ExperimentTemplate(
                name='TestExperiment',
                seed=42,
                model=Model(name='GCN'),
                dataset=Dataset(name='Cora'),
                attack=Attack(scope='global', type='evasion', name='DICE', epsilon=0.1),
                training=Training(max_epochs=200, patience=20),
                optimizer=Optimizer(name='Adam'),
                loss=Loss(name='CE')
            )
        ]
    )
    # Check that the configuration is valid and doesn't throw any exceptions
    assert valid_config.output_dir == './output'

@pytest.mark.parametrize("train_ratio, val_ratio, test_ratio, expected_train, expected_val, expected_test", [
    (0.7, 0.15, None, 0.7, 0.15, 0.15),  # Two ratios provided, test ratio calculated
    (None, 0.2, 0.7, 0.1, 0.2, 0.7),  # Two ratios provided, train ratio calculated
    (0.6, None, 0.3, 0.6, 0.1, 0.3),  # Two ratios provided, val ratio calculated
    (0.5, 0.3, 0.2, 0.5, 0.3, 0.2),  # All three ratios provided, sum less than 1
])
def test_check_ratios_valid(train_ratio, val_ratio, test_ratio, expected_train, expected_val, expected_test):
    result_train, result_val, result_test = check_ratios(train_ratio, val_ratio, test_ratio)
    assert result_train == pytest.approx(expected_train)
    assert result_val == pytest.approx(expected_val)
    assert result_test == pytest.approx(expected_test)

@pytest.mark.parametrize("train_ratio, val_ratio, test_ratio, error_msg", [
    (0.8, 0.3, None, "train_ratio, val_ratio, and test_ratio sum must be less or equal to 1.0 but the sum is 1.1."),  # Two ratios provided, sum greater than 1
    (None, 0.9, 0.2, "train_ratio, val_ratio, and test_ratio sum must be less or equal to 1.0 but the sum is 1.1."),  # Two ratios provided, sum greater than 1
    (0.7, None, 0.4, "train_ratio, val_ratio, and test_ratio sum must be less or equal to 1.0 but the sum is 1.1."),  # Two ratios provided, sum greater than 1
    (0.5, 0.3, 0.3, "train_ratio, val_ratio, and test_ratio sum must be less or equal to 1.0 but the sum is 1.1."),  # All three ratios provided, sum greater than or equal to 1
    (0.6, None, None, "At least two of train_ratio, val_ratio, and test_ratio must be provided if any are given."),  # Only one ratio provided
    (None, 0.2, None, "At least two of train_ratio, val_ratio, and test_ratio must be provided if any are given."),  # Only one ratio provided
    (None, None, 0.8, "At least two of train_ratio, val_ratio, and test_ratio must be provided if any are given."),  # Only one ratio provided
])
def test_check_ratios_invalid(train_ratio, val_ratio, test_ratio, error_msg):
    with pytest.raises(ValueError) as excinfo:
        check_ratios(train_ratio, val_ratio, test_ratio)
    assert error_msg in str(excinfo.value)