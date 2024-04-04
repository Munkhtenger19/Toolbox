from yacs.config import CfgNode as CN
from config_defaults import parse_args
from config_def import cfg
import yaml
import torch

# from register import registry


args = parse_args()

cfg.merge_from_file(args.cfg_file)

print(cfg)
# print(registry)
# print(registry['optimizer']['adam'])

def train_model(cfg):
    # Instantiate the GNN model
    model = GNN(cfg)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    # Set up the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Load the data
    data = load_data(cfg)

    # Train the model
    for epoch in range(cfg.TRAIN.EPOCHS):
        for batch in data:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()

# def load_data(cfg):