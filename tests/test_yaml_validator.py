import pytest
import os
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
)


@pytest.fixture
def valid_config_path():
    return os.path.join(os.path.dirname(__file__), "test_configs", "valid_config.yaml")


@pytest.fixture
def invalid_config_path():
    return os.path.join(
        os.path.dirname(__file__), "test_configs", "invalid_config.yaml"
    )


def test_load_and_validate_yaml_valid(valid_config_path):
    config = load_and_validate_yaml(valid_config_path)
    assert isinstance(config, dict)
    assert "output_dir" in config
    assert "experiment_templates" in config


def test_load_and_validate_yaml_invalid(invalid_config_path, caplog):
    with pytest.raises(ValueError) as excinfo:
        load_and_validate_yaml(invalid_config_path)
    assert "Validation error(s) encountered" in str(excinfo.value)
    assert len(caplog.records) > 0
    for record in caplog.records:
        assert record.levelname == "ERROR"


def test_config_model():
    # Valid model configuration
    valid_model = Model(name="GCN", params={"hidden_channels": 64, "dropout": 0.5})
    assert valid_model.name == "GCN"
    assert valid_model.params == {"hidden_channels": 64, "dropout": 0.5}

    # Invalid model name
    with pytest.raises(ValueError) as excinfo:
        Model(name="InvalidModelName")
    assert "Invalid model name" in str(excinfo.value)
