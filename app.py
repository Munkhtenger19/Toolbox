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
available_optimizers = list(registry['optimizer'].keys())
available_losses = list(registry['loss'].keys())
available_transforms = list(registry['transform'].keys())

selected_models = st.sidebar.multiselect("Model", available_models, default=[m for m in ["GPRGNN", "AirGNN"] if m in available_models])
selected_datasets = st.sidebar.multiselect("Dataset", available_datasets, default=[d for d in ["Cora"] if d in available_datasets])
attack_scope = st.sidebar.selectbox("Attack Scope", ["None", "Global", "Local"], index=1)

selected_attacks = []
attack_strategies = ["poison"] # Default

if attack_scope != "None":
    # Filter attacks based on type if possible, or show all
    selected_attacks = st.sidebar.multiselect("Attack Method", available_attacks, default=[a for a in ["PRBCD", "DICE"] if a in available_attacks])
    attack_strategies = st.sidebar.multiselect("Attack Strategy", ["poison", "evasion"], default=["poison"])

# 2. Hyperparameters
st.sidebar.subheader("Training Params")
epochs = st.sidebar.number_input("Max Epochs", min_value=1, value=70)
patience = st.sidebar.number_input("Patience", min_value=1, value=50)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=1)

st.sidebar.subheader("Optimizer & Loss")
selected_optimizers = st.sidebar.multiselect("Optimizer", available_optimizers, default=[o for o in ["Adam"] if o in available_optimizers])
learning_rate = st.sidebar.number_input("Learning Rate", value=0.01, format="%.4f")
weight_decay = st.sidebar.number_input("Weight Decay", value=0.0005, format="%.5f")

selected_losses = st.sidebar.multiselect("Loss Function", available_losses, default=["CE"] if "CE" in available_losses else [available_losses[0]])

st.sidebar.subheader("Data Transforms")
selected_transforms = st.sidebar.multiselect("Select Transforms", available_transforms, default=[t for t in ["NormalizeFeatures"] if t in available_transforms])
make_undirected = st.sidebar.checkbox("Make Undirected", value=False)
binary_attr = st.sidebar.checkbox("Binary Attributes", value=False)

st.sidebar.subheader("Data Split")
train_ratio = st.sidebar.slider("Train Ratio", 0.0, 1.0, 0.1)
val_ratio = st.sidebar.slider("Validation Ratio", 0.0, 1.0, 0.1)
test_ratio = st.sidebar.slider("Test Ratio", 0.0, 1.0, 0.8)

st.sidebar.subheader("Experiment Settings")
seeds_input = st.sidebar.text_input("Seeds (comma separated)", value="0")
seeds = [int(s.strip()) for s in seeds_input.split(",") if s.strip().isdigit()]
resume_output = st.sidebar.checkbox("Resume Output", value=True)
csv_save = st.sidebar.checkbox("Save CSV", value=True)

# Attack Params
attack_params = {}
epsilon = 0.05
target_nodes = []

if attack_scope != "None" and selected_attacks:
    st.sidebar.subheader("Attack Params")
    epsilon_input = st.sidebar.text_input("Perturbation Budget (epsilon, comma separated)", value="0.05")
    epsilon = [float(e.strip()) for e in epsilon_input.split(",") if e.strip()]
    
    if attack_scope == "Local":
        target_nodes_input = st.sidebar.text_input("Target Nodes (comma separated)", value="0")
        target_nodes = [int(n.strip()) for n in target_nodes_input.split(",") if n.strip().isdigit()]

    if "DICE" in selected_attacks:
        add_ratio = st.sidebar.slider("Add Ratio (DICE)", 0.0, 1.0, 0.6, 0.1)
        attack_params["add_ratio"] = add_ratio
    
    if any(a in selected_attacks for a in ["PRBCD", "GRBCD"]):
        block_size = st.sidebar.number_input("Block Size (PRBCD/GRBCD)", min_value=100, value=200000, step=1000)
        attack_epochs = st.sidebar.number_input("Attack Epochs (PRBCD/GRBCD)", min_value=1, value=80)
        attack_params["block_size"] = block_size
        attack_params["epochs"] = attack_epochs

# --- Main Area: Config Preview & Execution ---

# Construct the configuration dictionary dynamically
# This mimics the structure found in configs/default_experiment.yaml
experiment_config = {
    "output_dir": "./output_ui",
    "cache_dir": "./cache_ui",
    "resume_output": resume_output,
    "csv_save": csv_save,
    "experiment_templates": [
        {
            "name": "Streamlit_Experiment",
            "seed": seeds,
            "device": device,
            "model": {
                "name": selected_models,
                "params": {
                    # "in_channels": 1433, # Removed: Inferred dynamically by prepare_dataset
                    "hidden_channels": 64,
                    # "out_channels": 7,   # Removed: Inferred dynamically by prepare_dataset
                    "dropout": 0.5
                }
            },
            "dataset": {
                "name": selected_datasets,
                "root": "./datasets",
                "transforms": [{"name": t, "params": {"value": 0.8}} if t == "Constant" else {"name": t} for t in selected_transforms],
                "make_undirected": make_undirected,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio
            },
            "training": {
                "max_epochs": int(epochs),
                "patience": int(patience)
            },
            "optimizer": {
                "name": selected_optimizers,
                "params": {"lr": learning_rate, "weight_decay": weight_decay}
            },
            "loss": {
                "name": selected_losses
            }
        }
    ]
}

# Add attack config if selected
if attack_scope != "None" and selected_attacks:
    attack_conf = {
        "name": selected_attacks,
        "scope": attack_scope.lower(),
        "type": attack_strategies,
        "epsilon": epsilon,
        "binary_attr": binary_attr,
        **attack_params
    }
    if attack_scope == "Local":
        attack_conf["nodes"] = target_nodes

    experiment_config["experiment_templates"][0]["attack"] = attack_conf

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
                    clean_res = result.get('clean_result', [])
                    clean_acc = 'N/A'
                    if isinstance(clean_res, list) and clean_res:
                        clean_acc = clean_res[-1].get('Test accuracy after the best model retrieval', 'N/A')

                    perturbed_acc = 'N/A'
                    pert_res = result.get('perturbed_result', [])
                    if isinstance(pert_res, list) and pert_res:
                         perturbed_acc = pert_res[-1].get('Test accuracy after the best model retrieval', 'N/A')

                    attack_name = experiment.get('attack', {}).get('name', 'None')
                    epsilon = experiment.get('attack', {}).get('epsilon', 'N/A')
                    attack_type = experiment.get('attack', {}).get('type', 'N/A')
                    seed = experiment.get('seed', 'N/A')

                    results_list.append({
                        "Experiment": experiment['name'],
                        "Model": experiment['model']['name'],
                        "Dataset": experiment['dataset']['name'],
                        "Attack": attack_name,
                        "Type": attack_type,
                        "Epsilon": epsilon,
                        "Seed": seed,
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