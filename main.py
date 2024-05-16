import logging

from gnn_toolbox.cmd_args import parse_args, logger_setup
from gnn_toolbox.experiment_handler.exp_gen import generate_experiments_from_yaml
from gnn_toolbox.experiment_handler.exp_runner import run_experiment
from gnn_toolbox.experiment_handler.result_saver import LogExperiment
from gnn_toolbox.experiment_handler.artifact_manager import ArtifactManager
from gnn_toolbox.experiment_handler.config_validator import load_and_validate_yaml

def main(file):
    try:
        experiments_config = load_and_validate_yaml(file)
        experiments = generate_experiments_from_yaml(experiments_config)
        artifact_manager = ArtifactManager('cache2')
        logging.info(f'Running {len(experiments)} experiments')
        for curr_dir, experiment in experiments.items():  
            result, experiment_cfg = run_experiment(experiment, curr_dir, artifact_manager)
            experiment_logger = LogExperiment(curr_dir, experiment_cfg, result)
            experiment_logger.save_results()
    except Exception as e:
        logging.error(f"Failed to run the experiments: {e}")
    else:
        logging.info('Finished running the experiments.')

if __name__ =='__main__':
    args = parse_args()
    logger_setup(args.log)
    main(args.cfg)