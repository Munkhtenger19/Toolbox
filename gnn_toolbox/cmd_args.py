import os
import argparse
import logging

def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='Run experiments from a configuration file.')

    parser.add_argument('--cfg', type=str, default= os.path.join('configs', 'default_experiment.yaml'), help='The configuration YAML file path. Default is configs/good_1.yaml.')
    
    parser.add_argument('--log', type=str, default='INFO', help='Logging level. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL.')
    
    return parser.parse_args()

def logger_setup(logging_level):
    """
    Setup the logger for the experiment. Modified from https://docs.python.org/3/howto/logging.html
    """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    logger = logging.getLogger()
    logging_level = levels.get(logging_level.upper(), logging.INFO)
    logger.setLevel(logging_level)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)