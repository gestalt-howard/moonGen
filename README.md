# moonGen
Classifying difficulty of climbing routes on the MoonBoard apparatus using novel machine learning / deep learning techniques

This is the project repository for the CS230 Winter 2020 course project

## Running Experiments
For all experiments, make sure to change the root directory indicated in each script!

To run baseline experiments:
1. First, generate data for baseline experiments by running `\scripts\baseline\gen_baseline_data.py`
2. Batch-run baseline experiments by running `\scripts\baseline\run_baseline_models.py`

To run neural network experiments, simply execute `\scripts\pytorch\run_pytorch_models.py` (after making sure that the root directories are proper)

## Current Status (DONE)
Completely finished:
* Scraping
* Baseline PyTorch framework: Dense (Fully-Connected), GCN
* Baseline statistical learning models
* Evaluation metrics
* Batch-run of all experiments

## Potential Future Items
* Convolutional neural network as another baseline (work off of problem-hold matrix)
* Autoencoder for hold embedding extraction
