import pytest
from gnn_toolbox.registration_handler.registry import (
    registry,
    register_component,
    get_from_registry,
    check_model_signature,
)
from gnn_toolbox.registration_handler.register_components import (
    register_model,
    register_global_attack,
    register_local_attack,
    register_dataset,
    register_transform,
    register_optimizer,
    register_loss,
)


############### Mock classes and fixture for arrange step ###############

class MockModel:
    def forward(self, x, edge_index):
        pass

class MockModelWithKeywordArgs:
    def forward(self, x, edge_index, **kwargs):
        pass

class MockModelWithEdgeWeight:
    def forward(self, x, edge_index, edge_weight):
        pass

class MockModelWithEdgeWeightAndKeywordArgs:
    def forward(self, x, edge_index, edge_weight, **kwargs):
        pass

class MockModelWithInvalidParam:
    def forward(self, x, edge_index, invalid_param):
        pass

class MockModelWithoutForward:
    pass

@pytest.fixture()
def reset_registry():
    """Clear the registry before each test."""
    for category in registry:
        registry[category].clear()


############### register_component tests ###############

@pytest.mark.parametrize("category, key, component", [
    ("model", "TestModel", MockModel),
    ("dataset", "TestDataset", lambda x: x), # lambda x: x is a dummy function
    ("global_attack", "TestAttack", lambda x: x),
])
def test_register_component(reset_registry, category, key, component, ):
    register_component(category, key, component)
    assert registry[category][key] == component
    assert key in registry[category]
    

def test_register_component_invalid_category(reset_registry):
    with pytest.raises(ValueError) as e:
        register_component("unknown_category", "TestComponent", lambda x: x)
        
    assert "Category 'unknown_category' is not valid" in str(e.value)

def test_register_component_duplicate_key(reset_registry):
    register_component("model", "TestModel", MockModel)
    with pytest.raises(KeyError) as e:
        register_component("model", "TestModel", MockModelWithKeywordArgs)
    assert "Component with 'TestModel' already defined in category 'model'" in str(e.value)


############### get_from_registry tests ###############

@pytest.mark.parametrize("category, key, component", [
    ("model", "TestModel", MockModel),
    ("dataset", "TestDataset", lambda x: x),
    ("global_attack", "TestAttack", int),
])
def test_get_from_registry(reset_registry, category, key, component):
    register_component(category, key, component)
    retrieved_component = get_from_registry(category, key, registry)
    assert retrieved_component == component

def test_get_from_registry_invalid_category(reset_registry):
    with pytest.raises(ValueError) as e:
        get_from_registry("invalid_category", "TestComponent", registry)
    assert "Category 'invalid_category' is not recognized" in str(e.value)

def test_get_from_registry_nonexistent_key(reset_registry):
    with pytest.raises(KeyError) as e:
        get_from_registry("model", "UnexistingModel", registry)
    assert "Component 'UnexistingModel' not found in category 'model'" in str(e.value)


############### Registration functions tests ###############
    
@pytest.mark.parametrize("registration_function, category", [
    (register_global_attack, "global_attack"),
    (register_local_attack, "local_attack"),
    (register_dataset, "dataset"),
    (register_transform, "transform"),
    (register_optimizer, "optimizer"),
    (register_loss, "loss"),
])
def test_individual_registration_functions(reset_registry, registration_function, category):
    test_input = 'test_input'
    expected_output = test_input  # The lambda function simply returns its input
    registration_function("TestComponent", lambda x: x)
    assert "TestComponent" in registry[category]
    
    # Call the registered function and compare its output
    registered_function = registry[category]["TestComponent"]
    assert registered_function(test_input) == expected_output

############### check_model_signature tests ###############

@pytest.mark.parametrize("model_class", [
    MockModel,
    MockModelWithKeywordArgs,
    MockModelWithEdgeWeight,
    MockModelWithEdgeWeightAndKeywordArgs,
])
def test_check_model_signature_valid(reset_registry, model_class):
    # These models have valid signatures
    check_model_signature(model_class)

def test_check_model_signature_invalid_param(reset_registry):
    with pytest.raises(TypeError) as e:
        check_model_signature(MockModelWithInvalidParam)
    assert "Invalid parameter 'invalid_param'" in str(e.value)

def test_check_model_signature_no_forward(reset_registry):
    with pytest.raises(TypeError) as e:
        check_model_signature(MockModelWithoutForward)
    assert "The class MockModelWithoutForward or its ancestors do not define a 'forward' method." in str(e.value)