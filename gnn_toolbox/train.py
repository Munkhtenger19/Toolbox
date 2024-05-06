def train_and_evaluate(model, train_attr, train_adj, test_attr, test_adj, labels, split, device, writer, current_config, retrain, is_unattacked_model):
    # Move data to the device (GPU or CPU)
    train_attr = train_attr.to(device)
    train_adj = train_adj.to(device)
    test_attr = test_attr.to(device)
    test_adj = test_adj.to(device)
    labels = labels.to(device)
    
    if(retrain):
        model = copy.deepcopy(model).to(device)
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    # check for the model        
    
    # * make copy of optimizer
    
    
    # Train the model
    result = train(model=model, attr=train_attr, adj=train_adj, labels=labels, idx_train=split['train'], idx_val=split['valid'], idx_test=split['test'], writer= writer, **current_config.training, **current_config.optimizer.params)

    # Evaluate the model
    _, accuracy = evaluate_global(model=model, attr=test_attr, adj=test_adj, labels=labels, idx_test=split['test'], device=device)
    
    
        
    result.append({
        'Test accuracy after best model retrieval': accuracy
    })

    artifact_manager.save_model(model, current_config, result, is_unattacked_model)
    
    return result