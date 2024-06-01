## GNN Robustness Toolbox (GRT)

The GNN Robustness Toolbox (GRT) is a Python framework for evaluating the robustness of Graph Neural Network (GNN) models against adversarial attacks. GRT provides a flexible and extensible platform for conducting robustness experiments, enabling researchers and practitioners to:

* Systematically evaluate the robustness of different GNN architectures.
* Compare the effectiveness of various adversarial attack strategies.
* Develop and benchmark defense mechanisms against adversarial attacks.

**Key Features:**

* **Extensible Architecture:**  Easily integrate custom models, attacks, datasets, transforms, optimizers, and loss functions.
* **Flexible Configuration:** Define experiments using a user-friendly YAML configuration file.
* **Model Caching:**  Cache trained models and results to avoid redundant computations.
* **Comprehensive Output:**  Generate detailed experiment results in JSON and CSV format.

**Installation:**

```bash
git clone https://github.com/Munkhtenger19/Toolbox.git
cd Toolbox
pip install -r requirements.txt
```

**Usage:**

1. **Define Custom Components (Optional):** Create and register custom models, attacks, datasets, etc., in the `custom_components` directory.
2. **Configure Experiments:**  Create a YAML configuration file specifying the experiment settings (see `configs/` for examples).
3. **Run Experiments:**  Execute the `main.py` script with the configuration file path:

   ```bash
   python main.py --cfg path/to/config.yaml
   ```


**License:**

GRT is released under the [MIT License](LICENSE).