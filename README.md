# Gradient-Based Optimization for Handwritten Digit Classification (MNIST)

This project focuses on the C-based implementation of a multi-layer perceptron (MLP) neural network optimized using various gradient-based optimization techniques. The project explores how neural networks work at a fundamental level, including forward propagation, backpropagation, and different optimization strategies implemented from scratch. The goal is to classify handwritten digits from the MNIST dataset using techniques like **Stochastic Gradient Descent (SGD)**, **Learning Rate Decay**, **Momentum**, and **Adam**.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Files Description](#files-description)
- [Understanding the Neural Network and Backpropagation](#understanding-the-neural-network-and-backpropagation)
- [Implementation Details](#implementation-details)
  - [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
  - [Learning Rate Decay](#learning-rate-decay)
  - [Momentum](#momentum)
  - [Adaptive Learning Methods (Adam)](#adaptive-learning-methods-adam)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

## Project Overview

This project implements a multi-layer perceptron (MLP) neural network from scratch in C. The MLP is trained on the MNIST dataset for handwritten digit classification. We focus on understanding how neural networks compute predictions through forward propagation, how they learn through backpropagation, and how various optimization algorithms can improve convergence and accuracy. All these algorithms are implemented manually, providing an in-depth understanding of how each works at a low level.

## Features

- Full C implementation of a Multi-Layer Perceptron (MLP).
- Manual implementation of optimization techniques:
  - Stochastic Gradient Descent (SGD)
  - Learning Rate Decay
  - Momentum
  - Adaptive Learning Methods (Adam)
- Detailed forward propagation and backpropagation implementation for training the network.
- Training and testing on the MNIST dataset.
- Report generation with accuracy and loss metrics.

## Installation

To set up and run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gradient-based-optimization.git
   cd gradient-based-optimization
   ```

2. Install the necessary tools:
   - **GCC compiler**: The project is implemented in C, so ensure that you have GCC installed:
     ```bash
     sudo apt-get install gcc
     ```

3. Download the **MNIST dataset**, unzip it, and place it in the project directory.

## Usage

1. Compile the source code using the provided `Makefile`:
   ```bash
   make
   ```

2. Run the program with your choice of learning rate, batch size, and number of epochs:
   ```bash
   ./mnist_optimiser.out <path_to_mnist_data> <learning_rate> <batch_size> <num_epochs>
   ```

   Example:
   ```bash
   ./mnist_optimiser.out ./mnist_data 0.01 10 100
   ```

## Files Description

- `main.c`: Initializes the dataset, the neural network, and the optimizer, then starts the training process.
- `optimiser.c`: Implements the optimization framework. Includes implementations for SGD, learning rate decay, momentum, and Adam, all from scratch.
- `mnist_helper.c`: Loads and parses the MNIST dataset into a format suitable for training and testing.
- `neural_network.c`: Defines how the neural network performs forward propagation and backpropagation. This file contains the core logic for neuron activations and weight updates.
- `neural_network.h`: Defines data structures for neurons, weight matrices, and gradients.
- `optimiser.h`: Header file that declares the optimizer functions used in `optimiser.c`.
- `mnist_helper.h`: Helper functions for dataset handling.
- `Makefile`: Provides the instructions for compiling the project.

## Understanding the Neural Network and Backpropagation

### Neural Network Architecture

The MLP in this project consists of:
- **Input layer**: 784 neurons (28x28 pixels from the MNIST dataset).
- **Three hidden layers**: The first with 300 neurons, followed by two layers with 100 neurons each.
- **Output layer**: 10 neurons corresponding to the digits 0-9.

### Forward Propagation

During forward propagation, the input (a flattened 28x28 pixel image) is passed through each layer of the network, where each neuron applies a weighted sum of its inputs and a non-linear activation function (ReLU for hidden layers, softmax for the output layer). The forward pass computes the network's prediction for the input sample.

### Backpropagation

Backpropagation is used to train the neural network by calculating the gradient of the loss function with respect to each weight. The gradients are then used to update the weights. Backpropagation works by computing the loss at the output layer and propagating this error backward through the network, layer by layer, using the chain rule to update each weight.

## Implementation Details

### Stochastic Gradient Descent (SGD)

SGD is the simplest optimization algorithm implemented. It updates the weights by subtracting a fraction of the gradient of the loss with respect to the weights:
```math
w = w - \eta \cdot \nabla L(w)
```
Where:
- \( w \) is the weight,
- \( \eta \) is the learning rate,
- \( \nabla L(w) \) is the gradient of the loss with respect to the weight.

### Learning Rate Decay

Learning rate decay reduces the learning rate over time to fine-tune the optimization process. The learning rate decreases according to the following formula:
```math
\eta_t = \eta_0 \cdot (1 - \frac{t}{T})
```
Where:
- \( \eta_0 \) is the initial learning rate,
- \( t \) is the current epoch,
- \( T \) is the total number of epochs.

### Momentum

Momentum accelerates the convergence by adding a velocity term to the gradient updates. This term smooths out the gradient and allows the algorithm to escape local minima:
```math
v_t = \beta v_{t-1} + \eta \nabla L(w)
w = w - v_t
```
Where \( \beta \) is a hyperparameter that controls how much momentum is applied.

### Adaptive Learning Methods (Adam)

Adam is an advanced optimization algorithm that adapts the learning rate for each parameter based on estimates of the first and second moments of the gradients:
```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(w)
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla L(w)^2
w = w - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
```
Adam provides faster convergence by adjusting the learning rate for each parameter individually.

## How to Run

1. Make sure the MNIST dataset is in the correct directory.
2. Compile the code using `make`.
3. Run the program with the desired parameters for learning rate, batch size, and number of epochs.

## Results

I document all the result(training loss, accuracy) and the training process in a report check it here  -[report.pdf](report.pdf)

## References

- [1] Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," IEEE, 1998.
- [2] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," AISTATS, 2010.
- [3] D. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," arXiv, 2014.
