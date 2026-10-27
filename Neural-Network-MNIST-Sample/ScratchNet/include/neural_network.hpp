#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "layer.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>

/**
 * NeuralNetwork class - manages layers and orchestrates forward/backward pass
 * Uses Stochastic Gradient Descent (SGD) for training
 */
class NeuralNetwork {
private:
    std::vector<Layer> layers;
    size_t globalBatch = 0;
    bool emptyLocalBatch = false;
    std::string lossFunction;  // "mse" or "cross_entropy"

    /**
     * Compute loss gradient with respect to network output
     * @param predicted Network output (A for last layer)
     * @param expected Expected/target values
     * @return Gradient with respect to output
     */
    Matrix computeLossGradient(const Matrix& predicted, const Matrix& expected);

    /**
     * Recursive backward pass - processes layer and calls itself for previous layer
     * @param layerIndex Current layer index to process
     * @param dA Gradient from the layer above (output side)
     * @param learning_rate Learning rate for weight updates
     */
    void backwardRecursive(size_t layerIndex, const Matrix& dA, float learning_rate);

    /**
     * Compute MSE loss
     */
    float computeMSE(const Matrix& predicted, const Matrix& expected);

    /**
     * Compute Cross-Entropy loss
     */
    float computeCrossEntropy(const Matrix& predicted, const Matrix& expected);

public:
    /**
     * Default constructor
     */
    NeuralNetwork();

    /**
     * Add a layer to the network
     * @param layer Layer to add
     */
    void addLayer(const Layer& layer);

    /**
     * Set the loss function
     * @param loss "mse" for mean squared error, "cross_entropy" for cross-entropy
     */
    void setLossFunction(const std::string& loss);

    /**
     * Get number of layers
     */
    size_t getLayerCount() const;

    /**
     * Get layer by index
     */
    Layer& getLayer(size_t index);

    /**
     * Forward pass - pass input through all layers sequentially
     * @param input Input matrix (batch_size × input_size)
     * @return Output matrix (batch_size × output_size)
     */
    Matrix forward(const Matrix& input);

    /**
     * Backward pass - backpropagation algorithm
     * @param expected_output Target values
     * @param learning_rate Learning rate for weight updates
     */
    void backward(const Matrix& expected_output, float learning_rate);

    /**
     * Train the network using Stochastic Gradient Descent
     * @param X Training inputs (num_samples × input_size)
     * @param Y Training targets (num_samples × output_size)
     * @param epochs Number of epochs to train
     * @param learning_rate Learning rate for SGD
     */
    void train(const std::vector<Matrix>& X, const std::vector<Matrix>& Y, 
               int epochs, float learning_rate, size_t batchSize = 64, unsigned int seed = 42);

    /**
     * Compute loss for given predictions and targets
     */
    float computeLoss(const Matrix& predicted, const Matrix& expected);

    /**
     * Make predictions on new input
     * @param input Input matrix
     * @return Predicted output
     */
    Matrix predict(const Matrix& input);

    /**
     * Print network architecture
     */
    void printArchitecture() const;
};

#endif // NEURAL_NETWORK_HPP
