import csv
import json
import logging
from datetime import datetime
from pathlib import Path


class LogExperiment:
    def __init__(self, experiment_dir, experiment_cfg, result, csv_save):
        self.experiment_config = experiment_cfg
        self.result = result
        self.log_to = Path(experiment_dir)
        self.csv_save = csv_save 
        self.perturbed_result2csv = experiment_cfg['attack'].get('type') == 'poison'
    
    def save_results(self):
        self.log_experiment_config()
        self.log_results()
    
    def log_experiment_config(self):
        try:
            file_path = self.log_to / "experiment_config.json"
            with open(file_path, 'w') as f:
                json.dump(self.experiment_config, f, indent=4)
            logging.info(f"Experiment configuration logged to {file_path} successfully.")
        except Exception as e:
            logging.error(f"Failed to log experiment configuration to {file_path}: {e}")

    def log_results(self):
        try:
            for key, value in self.result.items():
                if value is not None:
                    file_path = self.log_to / f"{key}.json"
                    with open(file_path, 'w') as f:
                        json.dump(value, f, indent=2)
                    if self.csv_save and (key=='clean_result' or self.perturbed_result2csv):
                        self.save_to_csv(key, value)
                    # elif self.csv_save and key=='perturbed_result':
                    #    self.save_to_csv(key, value)
            logging.info(f"Results logged successfully to {self.log_to}.")
        except Exception as e:
            logging.error(f"Failed to log results to {self.log_to}: {e}")
            
    def save_to_csv(self, key, value):
        try:
            file_path = self.log_to / f"{key}.csv"
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(value[0].keys())
                for result in value:
                    writer.writerow(result.values())
            logging.info(f"{key} saved to CSV at {file_path} successfully.")
        except Exception as e:
            logging.error(f"Failed to save {key} to CSV at {file_path}: {e}")
            