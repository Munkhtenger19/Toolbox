import csv
import json
import logging
from datetime import datetime
from pathlib import Path


class LogExperiment:
    def __init__(self, experiment_dir, experiment_cfg, result):
        self.experiment_config = experiment_cfg
        self.result = result
        self.log_to = Path(experiment_dir)
        self.csv_save = experiment_cfg.get('csv_save', True) 
        self.perturbed_result2csv = experiment_cfg['attack'].get('type') == 'poison'
    
    def save_results(self):
        self.log_experiment_config()
        self.log_results()
    
    def log_experiment_config(self):
        try:
            file_path = self.log_to / "experiment_config.json"
            with open(file_path, 'w') as f:
                json.dump(self.experiment_config, f, indent=4)
            logging.info("Experiment configuration logged successfully.")
        except Exception as e:
            logging.error(f"Failed to log experiment configuration: {e}")

    def log_results(self):
        try:
            for key, value in self.result.items():
                if value is not None:
                    file_path = self.log_to / f"{key}.json"
                    with open(file_path, 'w') as f:
                        json.dump(value, f, indent=2)
                    if self.csv_save and (key=='clean_result' or self.perturbed_result2csv):
                        self.save_to_csv(key, value)
            logging.info("Results logged successfully.")
        except Exception as e:
            logging.error(f"Failed to log results: {e}")
            
    def save_to_csv(self, key, value):
        try:
            file_path = self.log_to / f"{key}.csv"
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(value[0].keys())
                for result in value:
                    writer.writerow(result.values())
            logging.info(f"{key} saved to CSV successfully.")
        except Exception as e:
            logging.error(f"Failed to save {key} to CSV: {e}")
            