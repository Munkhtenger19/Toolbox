from yacs.config import CfgNode as CN
from cmd_args import parse_args
from config_def import cfg
import yaml
from customize.train import train

from customize.util import create_model
from customize.train import BaseDataModule

args = parse_args()
cfg.merge_from_file(args.cfg_file)

loader = BaseDataModule()
model = create_model()
train(model, loader)
