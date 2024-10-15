

# Gradient-Based Optimization for Handwritten Digit Classification (MNIST)

This project implements a multi-layer perceptron (MLP) neural network optimized using various gradient-based optimization techniques, including **Stochastic Gradient Descent (SGD)**, **Learning Rate Decay**, **Momentum**, and **Adaptive Learning Methods** like **Adam** and **RMSProp**. The goal is to classify handwritten digits from the MNIST dataset with high accuracy and efficient convergence.

## Table of Contents
- [Gradient-Based Optimization for Handwritten Digit Classification (MNIST)](#gradient-based-optimization-for-handwritten-digit-classification-mnist)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Files Description](#files-description)
  - [Implementation Details](#implementation-details)
    - [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
    - [Learning Rate Decay](#learning-rate-decay)
    - [Momentum](#momentum)
    - [Adaptive Learning Methods (Adam)](#adaptive-learning-methods-adam)
  - [How to Run](#how-to-run)
  - [Results](#results)
  - [References](#references)

## Project Overview

The aim of this project is to implement and compare different optimization techniques to train a neural network for classifying handwritten digits using the **MNIST** dataset. The neural network is a multi-layer perceptron consisting of five layers:
- Input layer: 784 neurons (28x28 pixels)
- Three hidden layers: 300, 100, and 100 neurons respectively
- Output layer: 10 neurons (for digits 0-9)

We evaluate the performance of different optimization strategies based on their classification accuracy and computational efficiency.

## Features

- Multi-layer perceptron (MLP) neural network implementation.
- Optimization techniques: 
  - Stochastic Gradient Descent (SGD)
  - Learning Rate Decay
  - Momentum
  - Adaptive Learning Rate methods (Adam)
- Training and testing using the MNIST dataset.
- Detailed report generation with accuracy and loss visualization.
  
## Installation

Follow these steps to set up and run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gradient-based-optimization.git
   cd gradient-based-optimization
   ```

2. Download and install the necessary dependencies:
   - C compiler: You need `gcc` to compile the C code. If you don't have it installed, you can install it by running:
     ```bash
     sudo apt-get install gcc
     ```
   - Ensure the **MNIST dataset** files are available and unzip them in the same directory.

## Usage

1. Compile the source code:
   ```bash
   make
   ```

2. Run the program with your preferred hyperparameters:
   ```bash
   ./mnist_optimiser.out <path_to_mnist_data> <learning_rate> <batch_size> <num_epochs>
   ```

   Example:
   ```bash
   ./mnist_optimiser.out ./mnist_data 0.01 10 100
   ```

## Files Description

- `main.c`: Entry point of the program. Initializes the dataset, neural network, and optimizer. Starts the training process.
- `optimiser.c`: Implements the optimization framework, including SGD, learning rate decay, momentum, and Adam.
- `mnist_helper.c`: Contains helper functions for loading and parsing the MNIST dataset.
- `neural_network.c`: Defines the forward and backward pass of the neural network.
- `neural_network.h`: Header file for defining neuron structures and weight matrices.
- `optimiser.h`: Header file for the optimizer functions.
- `mnist_helper.h`: Header file for dataset loading functions.
- `Makefile`: Instructions for compiling the project.

## Implementation Details

### Stochastic Gradient Descent (SGD)

SGD updates the weights by calculating the gradient of the loss with respect to each weight and then subtracting this value, scaled by the learning rate. The update rule is:
```math
w = w - \eta \cdot \nabla L(w)
```
Where:
- \( w \) is the weight matrix,
- \( \eta \) is the learning rate,
- \( \nabla L(w) \) is the gradient of the loss function.

### Learning Rate Decay

To improve convergence, we use learning rate decay. The learning rate decreases over time as follows:
```math
\eta_t = \eta_0 \cdot (1 - \frac{t}{T})
```
Where \( t \) is the current epoch, and \( T \) is the total number of epochs.

### Momentum

Momentum helps accelerate learning by adding a velocity term:
```math
v_t = \beta v_{t-1} + \eta \nabla L(w)
w = w - v_t
```
Where \( \beta \) is a momentum term, typically between 0.5 and 0.9.

### Adaptive Learning Methods (Adam)

Adam adjusts the learning rate for each parameter dynamically using estimates of first and second moments:
```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(w)
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla L(w)^2
w = w - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
```

## How to Run

1. Ensure the MNIST dataset is available in the specified directory.
2. Compile the code using `make`.
3. Run the program with appropriate arguments for the dataset path, learning rate, batch size, and number of epochs.

## Results

The program will print the training loss and accuracy for each epoch. After training, it will display the final test accuracy. The results can be visualized using a plotting tool to track the convergence of the optimization algorithms.

## References

- [1] Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," IEEE, 1998.
- [2] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," AISTATS, 2010.
- [3] D. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," arXiv, 2014.