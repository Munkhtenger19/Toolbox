import os
import json
import logging
from datetime import datetime
from pathlib import Path

class LogExperiment:
    def __init__(self, experiment_dir, experiment_cfg, result):
        self.experiment_config = experiment_cfg
        self.result = result
        self.log_to = Path(experiment_dir)
        self.log_experiment_config()
        self.log_results()
        
    def log_experiment_config(self):
        file_path = self.log_to / "experiment_config.json"
        with open(file_path, 'w') as f:
            json.dump(self.experiment_config, f, indent=4)

    def log_results(self):
        for key, value in self.result.items():
            file_path = self.log_to / f"{key}.json"
            with open(file_path, 'w') as f:
                json.dump(value, f, indent=2)
            

def setup_logging(experiment):
    """ Set up the logging environment. """
    log_dir = f"./logs/{experiment_id}"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f"{log_dir}/experiment.log",
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    return log_dir

def log_experiment_results(experiment_config):
    output_dir = Path(experiment_config['name'])
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{experiment_config['name']}_results.json"
    
    with open(filename, 'w') as f:
        json.dump(experiment_config, f, indent=4)
        f.write("\n")
        # Additionally write out other results like accuracy
        f.write(json.dumps({'accuracy': accuracy}))