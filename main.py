import logging

from gnn_toolbox.cmd_args import parse_args, logger_setup
from gnn_toolbox.experiment_handler.exp_gen import generate_experiments_from_yaml
from gnn_toolbox.experiment_handler.exp_runner import run_experiment
from gnn_toolbox.experiment_handler.logger import LogExperiment
from gnn_toolbox.experiment_handler.artifact_manager import ArtifactManager

def main(file):
    # * here add pydantic checker for yaml
    # yaml = pydantic_checker(args.cfg_file)
    
    experiments = generate_experiments_from_yaml(file)
    artifact_manager = ArtifactManager('cache')
    try:
        for curr_dir, experiment in experiments.items():  
            result, experiment_cfg = run_experiment(experiment, curr_dir, artifact_manager)
            LogExperiment(curr_dir, experiment_cfg, result)
    except Exception as e:
        logging.error(f"Failed to run the experiments: {e}")
    else:
        logging.info('Finished running the experiments.')

if __name__ =='__main__':
    args = parse_args()
    logger_setup(args.log)
    main(args.cfg)