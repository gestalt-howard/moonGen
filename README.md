# moonGen
Classifying difficulty of climbing routes on the MoonBoard apparatus using novel machine learning / deep learning techniques

## Running Experiments
For all experiments, make sure to change the root directory indicated in each script!

To run baseline experiments:
1. First, generate data for baseline experiments by running `\scripts\baseline\gen_baseline_data.py`
2. Batch-run baseline experiments by running `\scripts\baseline\run_baseline_models.py`

To run neural network experiments, simply execute `\scripts\pytorch\run_pytorch_models.py` (after making sure that the root directories are proper)

## Current Status
Completely finished:
* Scraping
* Baseline PyTorch framework: Dense (Fully-Connected), GCN
* Baseline statistical learning models

Partially finished:
* Evaluation metrics

## Action Plan
### Low-Effort
* Adding in orientation features (likely not going to happen)
* Batch-running NN experiments

### Medium-Effort
* Re-defining PMI calculation in hold-hold adjacency relationship

### High-Effort
* Convolutional neural network as another baseline (work off of problem-hold matrix)
* Autoencoder for hold embedding extraction
