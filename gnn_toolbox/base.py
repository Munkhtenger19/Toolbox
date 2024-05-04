from yacs.config import CfgNode as CN
from cmd_args import parse_args
from config_def import cfg, set_run_dir
import yaml
import logging
from customize.train import train
from torch_geometric import seed_everything
from customize.utils import create_model, auto_select_device
from customize.train import BaseDataModule
from sacred import Experiment
ex = Experiment("Run_experiments")

from sacred.observers import MongoObserver, FileStorageObserver

ex.observers.append(FileStorageObserver('runs'))

@ex.main
def main():
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    auto_select_device()
    seed_everything(cfg.seed)
    set_run_dir(cfg.run_dir)
    loader = BaseDataModule()
    model = create_model()
    logging.info(model)
    logging.info(cfg)
    train(model, loader)
    # if evasion attack strategy after model trained and log the result
    attack = create_attack()
    attack.apply(model, data)
    # if poisoning, perturb the data and train the model on that data and log the result

if __name__ == '__main__':
    ex.run()
