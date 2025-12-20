import streamlit as st
import yaml
import pandas as pd
import logging
from pathlib import Path
import shutil

# Import internal modules
from gnn_toolbox.registry import registry
from gnn_toolbox.experiment_handler.exp_gen import generate_experiments_from_yaml
from gnn_toolbox.experiment_handler.exp_runner import run_experiment
from gnn_toolbox.experiment_handler.artifact_manager import ArtifactManager
from custom_components import *  # Ensure components are registered

# Configure Logging to show in Streamlit
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="GNN Robustness Toolbox", layout="wide")

st.title("🛡️ GNN Robustness Toolbox (GRT)")

# --- Sidebar: Configuration ---
st.sidebar.header("Experiment Configuration")

# 1. Select Components from Registry
available_models = list(registry['model'].keys())
available_datasets = list(registry['dataset'].keys())
available_attacks = list(registry['global_attack'].keys()) + list(registry['local_attack'].keys())

selected_model = st.sidebar.selectbox("Model", available_models, index=0)
selected_dataset = st.sidebar.selectbox("Dataset", available_datasets, index=0)
attack_type = st.sidebar.selectbox("Attack Type", ["None", "Global", "Local"], index=0)

selected_attack = None
if attack_type != "None":
    # Filter attacks based on type if possible, or show all
    selected_attack = st.sidebar.selectbox("Attack Method", available_attacks)

# 2. Hyperparameters
st.sidebar.subheader("Training Params")
epochs = st.sidebar.number_input("Max Epochs", min_value=1, value=100)
patience = st.sidebar.number_input("Patience", min_value=1, value=50)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)

# --- Main Area: Config Preview & Execution ---

# Construct the configuration dictionary dynamically
# This mimics the structure found in configs/default_experiment.yaml
experiment_config = {
    "output_dir": "./output_ui",
    "cache_dir": "./cache_ui",
    "resume_output": False,
    "csv_save": True,
    "experiment_templates": [
        {
            "name": f"{selected_model}_{selected_dataset}_Exp",
            "seed": [0],
            "device": device,
            "model": {
                "name": selected_model,
                "params": {
                    "in_channels": 1433, # Default for Cora, logic needed for dynamic sizing
                    "hidden_channels": 64,
                    "out_channels": 7,
                    "dropout": 0.5
                }
            },
            "dataset": {
                "name": selected_dataset,
                "root": "./datasets"
            },
            "training": {
                "max_epochs": int(epochs),
                "patience": int(patience)
            },
            "optimizer": {
                "name": "Adam",
                "params": {"lr": 0.01, "weight_decay": 5e-4}
            },
            "loss": {
                "name": "CE"
            }
        }
    ]
}

# Add attack config if selected
if attack_type != "None" and selected_attack:
    experiment_config["experiment_templates"][0]["attack"] = {
        "name": selected_attack,
        "scope": attack_type.lower(),
        "params": {"epsilon": 0.05} # Default param, could be expanded
    }

# Tabs for different views
tab1, tab2 = st.tabs(["Configuration", "Results"])

with tab1:
    st.subheader("Current Configuration")
    st.json(experiment_config)

    if st.button("Run Experiment", type="primary"):
        with st.spinner("Initializing Experiment..."):
            try:
                # 1. Generate Experiments
                experiments, cache_dir = generate_experiments_from_yaml(experiment_config)
                artifact_manager = ArtifactManager(cache_dir)
                
                results_list = []
                
                # 2. Run Loop
                progress_bar = st.progress(0)
                total_exps = len(experiments)
                
                for i, (curr_dir, experiment) in enumerate(experiments.items()):
                    st.write(f"Running: **{experiment['name']}**")
                    
                    # Execute logic from main.py
                    result, experiment_cfg = run_experiment(experiment, artifact_manager)
                    
                    # Extract key metrics for display
                    clean_acc = result.get('clean_result', [{}])[-1].get('accuracy_test', 'N/A')
                    perturbed_acc = 'N/A'
                    if 'perturbed_result' in result and result['perturbed_result']:
                         perturbed_acc = result['perturbed_result'].get('accuracy of the model', 'N/A')

                    results_list.append({
                        "Experiment": experiment['name'],
                        "Model": experiment['model']['name'],
                        "Dataset": experiment['dataset']['name'],
                        "Clean Accuracy": clean_acc,
                        "Perturbed Accuracy": perturbed_acc
                    })
                    
                    progress_bar.progress((i + 1) / total_exps)

                st.success("Experiment Completed!")
                
                # Store results in session state to persist across reruns
                st.session_state['results'] = pd.DataFrame(results_list)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

with tab2:
    st.subheader("Experiment Results")
    if 'results' in st.session_state:
        df = st.session_state['results']
        st.dataframe(df, use_container_width=True)
        
        # Simple visualization
        if "Clean Accuracy" in df.columns and "Perturbed Accuracy" in df.columns:
            # Filter out N/A for plotting
            plot_df = df[df["Perturbed Accuracy"] != 'N/A'].copy()
            if not plot_df.empty:
                st.bar_chart(plot_df.set_index("Experiment")[["Clean Accuracy", "Perturbed Accuracy"]])
    else:
        st.info("Run an experiment to see results here.")