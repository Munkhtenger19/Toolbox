import os
import argparse
import logging
from custom_components import *
from gnn_toolbox.registration_handler.registry import registry
def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description='GNN Robustness Toolbox: Run experiments to test GNNs against adversarial attacks from configuration YAML file.')

    parser.add_argument('--cfg', type=str, default= os.path.join('configs', 'default_experiment.yaml'), help='The configuration YAML file path. Default is configs/good_1.yaml.')
    
    parser.add_argument('--log', type=str, default='INFO', help='Logging level. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL.')
    
    parser.add_argument('--list-components', action='store_true', help="List registered components and exit")

    return parser.parse_args()

def list_registered_components():
    """
    List all registered components
    """
    for key, value in registry.items():
        print(f"Registered {key}: {list(value.keys())}")

def logger_setup(logging_level):
    """
    Setup the logger for the experiment. Modified from https://docs.python.org/3/howto/logging.html

    Args:
        logging_level (str): _description_
    """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger()
    logging_level = levels.get(logging_level.upper(), logging.INFO)
    logger.setLevel(logging_level)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)