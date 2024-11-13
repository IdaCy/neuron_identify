# Neuron Experimenting with Sparse Autoencoders

This repository contains code and tools for experimenting with isolating task-relevant neurons in multitask neural networks using sparse autoencoders.

## Usage

### 1. Setting Up the Environment

Install the required dependencies:

    pip install torch matplotlib seaborn notebook simpy

2. Generating Sample Data

To generate sample data for testing, run the sample_data_generation.py script:

    python3 data/sample_data_generation.py

3. Training the Autoencoders

To train sparse autoencoders on task-specific activations, run:

    python3 src/train_autoencoder.py

This will save the trained models in the models/ directory.

4. Visualizing Task-Specific Neurons

After training, you can run the Jupyter notebook notebooks/visualize_task_neurons.ipynb to visualise and run tests!

## Contributions

This project is open to contributions!
