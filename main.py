import logging
import sys

from gnn_toolbox.cmd_args import parse_args, logger_setup, list_registered_components
from gnn_toolbox.experiment_handler.exp_gen import generate_experiments_from_yaml
from gnn_toolbox.experiment_handler.exp_runner import run_experiment
from gnn_toolbox.experiment_handler.result_saver import LogExperiment
from gnn_toolbox.experiment_handler.artifact_manager import ArtifactManager
from gnn_toolbox.experiment_handler.config_validator import load_and_validate_yaml
from gnn_toolbox.experiment_handler.exceptions import DatasetCreationError, DataPreparationError, ModelCreationError, ModelTrainingError, GlobalAttackError, LocalAttackError

def main(file):
    try:
        experiments_config = load_and_validate_yaml(file)
        experiments, cache_dir = generate_experiments_from_yaml(experiments_config)
        artifact_manager = ArtifactManager(cache_dir)
        logging.info(f'Running {len(experiments)} experiment(s)')
        for curr_dir, experiment in experiments.items():
            try:
                result, experiment_cfg = run_experiment(experiment, curr_dir, artifact_manager)
                experiment_logger = LogExperiment(curr_dir, experiment_cfg, result)
                experiment_logger.save_results()
            except (DatasetCreationError, DataPreparationError,  ModelCreationError, ModelTrainingError) as e:
                # logging.exception(e)
                logging.error(f"Failed to run this experiment, so skipping to the next experiment: {experiment}. ")
                continue
            except (GlobalAttackError, LocalAttackError) as e:
                logging.error(f"Failed to run this experiment because of attack error, so skipping to the next experiment: {experiment}. ")
                continue
            except Exception as e:
                logging.exception(e)
                logging.error(f"Failed to run this experiment and save the result, so skipping to the next experiment: {experiment}.")
                continue
    except Exception as e:
        logging.error(f"There was an error loading YAML file or generating experiments from it.")
    else:
        logging.info('Finished running the experiments.')

if __name__ =='__main__':
    args = parse_args()
    if args.list_components:
        list_registered_components()
        sys.exit(0)
    logger_setup(args.log)
    main(args.cfg)