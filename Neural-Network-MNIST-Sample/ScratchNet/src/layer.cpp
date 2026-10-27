#include "layer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>

Layer::Layer()
    : inputSize(0), outputSize(0), activationName("relu") {
    initializeActivation("relu");
}

Layer::Layer(size_t inputSize, size_t outputSize, const std::string& activationName)
    : inputSize(inputSize), outputSize(outputSize), activationName(activationName) {
    initializeWeights();
    initializeActivation(activationName);
}

Layer::Layer(const Layer& other)
    : inputSize(other.inputSize),
      outputSize(other.outputSize),
      weights(other.weights),
      biases(other.biases),
      Z(other.Z),
      A(other.A),
      prevA(other.prevA),
      dZ(other.dZ),
      activationName(other.activationName) {
    initializeActivation(activationName);
}

Layer::~Layer() {}

// Static random engine to ensure different random values for each layer
static std::default_random_engine global_generator(std::random_device{}());

void Layer::setSeed(unsigned int seed) { global_generator.seed(seed); }

void Layer::initializeWeights() {
    // Xavier initialization: sqrt(1.0 / input_size)
    // Works better for tanh/sigmoid; for ReLU, He is technically better
    // but Xavier often works well enough and is more stable
    double scale = std::sqrt(1.0 / static_cast<double>(inputSize));
    std::normal_distribution<double> distribution(0.0, scale);
    
    std::vector<std::vector<double>> weightsData(outputSize, std::vector<double>(inputSize));
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weightsData[i][j] = distribution(global_generator);
        }
    }
    weights = Matrix(weightsData);
    // Initialize biases to small positive values to prevent dying ReLU
    std::uniform_real_distribution<double> bias_dist(0.0, 0.01);
    std::vector<std::vector<double>> biasesData(1, std::vector<double>(outputSize));
    for (size_t j = 0; j < outputSize; ++j) {
        biasesData[0][j] = bias_dist(global_generator);
    }
    biases = Matrix(biasesData);
}

void Layer::initializeActivation(const std::string& name) {
    if (name == "sigmoid") {
        activation = [](const Matrix& x) { return Activations::sigmoid(x); };
        activationDerivative = [](const Matrix& x) { return Activations::sigmoid_derivative(x); };
        activationInPlace = [](Matrix& x) { Activations::sigmoid_inplace(x); };
    } else if (name == "relu") {
        activation = [](const Matrix& x) { return Activations::relu(x); };
        activationDerivative = [](const Matrix& x) { return Activations::relu_derivative(x); };
        activationInPlace = [](Matrix& x) { Activations::relu_inplace(x); };
    } else if (name == "leaky_relu") {
        activation = [](const Matrix& x) { return Activations::leaky_relu(x); };
        activationDerivative = [](const Matrix& x) { return Activations::leaky_relu_derivative(x); };
        activationInPlace = [](Matrix& x) { Activations::leaky_relu_inplace(x); };
    } else if (name == "softmax") {
        activation = [](const Matrix& x) { return Activations::softmax(x); };
        // For softmax with cross-entropy, derivative is handled specially in backward pass
        // We don't need the activation derivative since dL/dZ = dL/dA for cross-entropy + softmax
        activationDerivative = [](const Matrix& x) { 
            // Return ones - actual gradient is computed in backward pass
            return x.apply([](double) { return 1.0; }); 
        };
        activationInPlace = [](Matrix& x) { 
            // Softmax needs special handling - apply in-place by modifying directly
            size_t rows = x.getRows();
            size_t cols = x.getCols();
            auto& data = x.getData();
            for (size_t i = 0; i < rows; ++i) {
                double max_val = data[i][0];
                for (size_t j = 1; j < cols; ++j) {
                    if (data[i][j] > max_val) max_val = data[i][j];
                }
                double sum_exp = 0.0;
                for (size_t j = 0; j < cols; ++j) {
                    data[i][j] = std::exp(data[i][j] - max_val);
                    sum_exp += data[i][j];
                }
                for (size_t j = 0; j < cols; ++j) {
                    data[i][j] /= sum_exp;
                }
            }
        };
    } else if (name == "tanh") {
        activation = [](const Matrix& x) { return Activations::tanh(x); };
        activationDerivative = [](const Matrix& x) { return Activations::tanh_derivative(x); };
        activationInPlace = [](Matrix& x) { Activations::tanh_inplace(x); };
    } else if (name == "linear") {
        activation = [](const Matrix& x) { return Activations::linear(x); };
        activationDerivative = [](const Matrix& x) { return Activations::linear_derivative(x); };
        activationInPlace = [](Matrix& x) { Activations::linear_inplace(x); };
    } else {
        throw std::invalid_argument("Unknown activation: " + name);
    }
    activationName = name;
}

Matrix Layer::forward(const Matrix& X) {
    prevA = X;
    Z = X.dot(weights.transpose());
    
    // Optimized bias addition - use direct data access
    size_t batchSize = Z.getRows();
    size_t biasCols = biases.getCols();
    auto& Z_data = Z.getData();
    const auto& bias_data = biases.getData();
    
    for (size_t i = 0; i < batchSize; ++i) {
        for (size_t j = 0; j < biasCols; ++j) {
            Z_data[i][j] += bias_data[0][j];
        }
    }
    
    // Use in-place activation for optimization
    A = Z;  // Copy Z to A (will be modified in-place)
    activationInPlace(A);
    return A;
}

Matrix Layer::getWeights() const { return weights; }
Matrix Layer::getBiases() const { return biases; }
Matrix Layer::getZ() const { return Z; }
Matrix Layer::getA() const { return A; }
Matrix Layer::getPrevA() const { return prevA; }
Matrix Layer::getDZ() const { return dZ; }
ActivationDerivative Layer::getActivationDerivative() const { return activationDerivative; }
ActivationInPlace Layer::getActivationInPlace() const { return activationInPlace; }
void Layer::setDZ(const Matrix& newDZ) { dZ = newDZ; }
size_t Layer::getInputSize() const { return inputSize; }
size_t Layer::getOutputSize() const { return outputSize; }
std::string Layer::getActivationName() const { return activationName; }

void Layer::setWeights(const Matrix& newWeights) {
    if (newWeights.getRows() != outputSize || newWeights.getCols() != inputSize) {
        throw std::invalid_argument("Weight dimensions mismatch");
    }
    weights = newWeights;
}

void Layer::setBiases(const Matrix& newBiases) {
    if (newBiases.getRows() != 1 || newBiases.getCols() != outputSize) {
        throw std::invalid_argument("Bias dimensions mismatch");
    }
    biases = newBiases;
}

void Layer::print() const {
    std::cout << "Layer: " << inputSize << " -> " << outputSize 
              << ", " << activationName << "\n";
    std::cout << "Weights:\n";
    weights.print();
    std::cout << "Biases:\n";
    biases.print();
}

Layer& Layer::operator=(const Layer& other) {
    if (this != &other) {
        inputSize = other.inputSize;
        outputSize = other.outputSize;
        weights = other.weights;
        biases = other.biases;
        Z = other.Z;
        A = other.A;
        prevA = other.prevA;
        dZ = other.dZ;
        activationName = other.activationName;
        initializeActivation(activationName);
    }
    return *this;
}
