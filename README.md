# Neuron Experimenting with Sparse Autoencoders

This repository contains code and tools for experimenting with isolating task-relevant neurons in multitask neural networks using sparse autoencoders. By focusing on sparse representations, we aim to improve mechanistic interpretability of neural networks by identifying neurons that are specific to individual tasks.

## Objectives

The main goals of this repository are:
- To use sparse autoencoders to learn compact representations for each task.
- To identify task-specific neurons in a multitask model, allowing us to isolate neurons that respond differently depending on the task.
- To provide an open framework for further research in neuron interpretability, where the community can contribute improvements.

## Usage

### 1. Setting Up the Environment

Install the required dependencies:

    pip install torch matplotlib

2. Generating Sample Data
To generate sample data for testing, run the sample_data_generation.py script:

    python3 data/sample_data_generation.py

3. Training the Autoencoders
To train sparse autoencoders on task-specific activations, run:

    python3 src/train_autoencoder.py

This will save the trained models in the models/ directory.

4. Visualizing Task-Specific Neurons

After training, you can run the Jupyter notebook notebooks/visualize_task_neurons.ipynb to visualize and analyze neuron activations for each task. This notebook includes histograms of activation distributions and allows for further analysis of task-specific neurons.


## How to Contribute

This project is open to contributions. If you have improvements, bug fixes, or new ideas, please feel free to submit a pull request.
