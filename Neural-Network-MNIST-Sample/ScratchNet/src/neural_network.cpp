#include "neural_network.hpp"
#include "parallel.hpp"
#include <cmath>
#include <algorithm>

NeuralNetwork::NeuralNetwork() : lossFunction("mse") {}


Matrix NeuralNetwork::computeLossGradient(const Matrix& predicted, const Matrix& expected) {
    size_t batchSize = predicted.getRows();
    
    if (lossFunction == "mse") {
        // MSE gradient: dL/dA = 2 * (predicted - expected) / n
        Matrix gradient = predicted - expected;
        return gradient * (2.0 / static_cast<double>(batchSize));
    } else if (lossFunction == "cross_entropy") {
        return predicted - expected;
    }
    
    // Default to MSE
    return predicted - expected;
}

float NeuralNetwork::computeMSE(const Matrix& predicted, const Matrix& expected) {
    size_t batchSize = predicted.getRows();
    size_t outputSize = predicted.getCols();
    
    double sum = 0.0;
    for (size_t i = 0; i < batchSize; ++i) {
        for (size_t j = 0; j < outputSize; ++j) {
            double diff = predicted.getElement(i, j) - expected.getElement(i, j);
            sum += diff * diff;
        }
    }
    
    return static_cast<float>(sum / (2.0 * batchSize));
}

float NeuralNetwork::computeCrossEntropy(const Matrix& predicted, const Matrix& expected) {
    size_t batchSize = predicted.getRows();
    size_t outputSize = predicted.getCols();
    
    double sum = 0.0;
    double epsilon = 1e-10;
    
    for (size_t i = 0; i < batchSize; ++i) {
        for (size_t j = 0; j < outputSize; ++j) {
            double p = predicted.getElement(i, j);
            double y = expected.getElement(i, j);
            p = std::max(epsilon, std::min(1.0 - epsilon, p));
            sum += y * std::log(p);
        }
    }
    
    return static_cast<float>(-sum / batchSize);
}


void NeuralNetwork::backwardRecursive(size_t layerIndex, const Matrix& dA, float learning_rate) {
    // Iterative backward pass (optimized - replaces recursion)
    Matrix dA_current = dA;
    
    for (size_t idx = layerIndex + 1; idx > 0; --idx) {
        size_t i = idx - 1;
        Layer& layer = layers[i];
        
        Matrix Z = layer.getZ();
        Matrix prevA = layer.getPrevA();
        Matrix W = layer.getWeights();
        size_t batchSize = Z.getRows();
        bool isOutputLayer = (i == layerIndex);
        
        // Special case: softmax output layer with cross-entropy loss
        // Gradient is already computed correctly in computeLossGradient
        // Don't apply activation derivative (it would give wrong results)
        Matrix dZ;
        if (isOutputLayer && lossFunction == "cross_entropy" && layer.getActivationName() == "softmax") {
            dZ = dA_current;
        } else {
            // dZ = dA * activation'(Z) (element-wise)
            ActivationDerivative activationDerivative = layer.getActivationDerivative();
            dZ = dA_current.multiply(activationDerivative(Z));
        }
        layer.setDZ(dZ);
        
        // dW = prevA.T * dZ (optimized: use dotTranspose to avoid explicit transpose)
        Matrix dW = prevA.dotTranspose(prevA, dZ).transpose();
        // Sum local gradients before dividing by the actual global batch size.
        
        // db = sum(dZ, axis=0) / batch_size (optimized: direct data access)
        Matrix db(1, dZ.getCols(), 0.0);
        const auto& dZ_data = dZ.getData();
        auto& db_data = db.getData();
        for (size_t j = 0; j < dZ.getCols(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < dZ.getRows(); ++k) {
                sum += dZ_data[k][j];
            }
            db_data[0][j] = sum;
        }
        
        parallel::exchange(dW, false);
        parallel::exchange(db, false);
        const double denominator = static_cast<double>(globalBatch ? globalBatch : batchSize);
        dW = dW.scalarMultiply(1.0 / denominator);
        db = db.scalarMultiply(1.0 / denominator);
        // Update weights and biases
        Matrix b = layer.getBiases();
        layer.setWeights(W - dW.scalarMultiply(learning_rate));
        layer.setBiases(b - db.scalarMultiply(learning_rate));
        
       
        if (i > 0) {
            dA_current = dZ.dot(W);
        }
    }
}


void NeuralNetwork::addLayer(const Layer& layer) {
    layers.push_back(layer);
}

void NeuralNetwork::setLossFunction(const std::string& loss) {
    lossFunction = loss;
}

size_t NeuralNetwork::getLayerCount() const {
    return layers.size();
}

Layer& NeuralNetwork::getLayer(size_t index) {
    return layers[index];
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix result = input;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        result = layers[i].forward(result);
    }
    
    return result;
}

void NeuralNetwork::backward(const Matrix& expected_output, float learning_rate) {
    if (layers.empty()) {
        return;
    }
    
    // Get output from last layer
    Matrix A_output = layers[layers.size() - 1].getA();
    
    // Compute initial gradient (dA) using loss function
    Matrix dA = computeLossGradient(A_output, expected_output);
    if (emptyLocalBatch) dA = Matrix(dA.getRows(), dA.getCols(), 0.0);
    
    // Start recursive backward pass from the last layer
    backwardRecursive(layers.size() - 1, dA, learning_rate);
}

// Helper function to stack matrices vertically (batch creation)
Matrix stackMatricesVertical(const std::vector<Matrix>& matrices) {
    if (matrices.empty()) return Matrix();
    
    size_t numSamples = matrices.size();
    size_t rows = matrices[0].getRows();
    size_t cols = matrices[0].getCols();
    
    std::vector<std::vector<double>> result(numSamples * rows, std::vector<double>(cols));
    
    for (size_t b = 0; b < numSamples; ++b) {
        const auto& data = matrices[b].getData();
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                result[b * rows + r][c] = data[r][c];
            }
        }
    }
    
    return Matrix(result);
}

void NeuralNetwork::train(const std::vector<Matrix>& X, const std::vector<Matrix>& Y,
                          int epochs, float learning_rate, size_t batchSize, unsigned int seed) {
    if (batchSize == 0 || X.empty() || X.size() != Y.size())
        throw std::invalid_argument("Invalid training data or batch size.");

    // Rank 0 chooses the image order. All laptops use that same order.
    std::vector<int> indices(X.size());
    for (size_t i = 0; i < X.size(); ++i) indices[i] = static_cast<int>(i);
    std::mt19937 generator(seed);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (parallel::rank == 0) std::shuffle(indices.begin(), indices.end(), generator);
#ifdef USE_MPI
        MPI_Bcast(indices.data(), static_cast<int>(indices.size()), MPI_INT, 0, MPI_COMM_WORLD);
#endif
        for (size_t start = 0; start < X.size(); start += batchSize) {
            globalBatch = std::min(batchSize, X.size() - start);
            // MPI splits each batch; sequential/OpenMP process the whole batch.
            size_t first = start + globalBatch * parallel::rank / parallel::size;
            size_t last = start + globalBatch * (parallel::rank + 1) / parallel::size;
            emptyLocalBatch = first == last;
            std::vector<Matrix> batchX, batchY;
            for (size_t i = first; i < last; ++i) {
                batchX.push_back(X[indices[i]]);
                batchY.push_back(Y[indices[i]]);
            }
            // An idle rank must still join MPI communication with zero gradients.
            if (emptyLocalBatch) {
                batchX.push_back(X[0]);
                batchY.push_back(Y[0]);
            }
            forward(stackMatricesVertical(batchX));       // Calculate predictions.
            backward(stackMatricesVertical(batchY), learning_rate); // Update weights.
        }
    }
}
float NeuralNetwork::computeLoss(const Matrix& predicted, const Matrix& expected) {
    if (lossFunction == "mse") {
        return computeMSE(predicted, expected);
    } else if (lossFunction == "cross_entropy") {
        return computeCrossEntropy(predicted, expected);
    }
    return computeMSE(predicted, expected);
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    return forward(input);
}

void NeuralNetwork::printArchitecture() const {
    std::cout << "Neural Network Architecture:" << std::endl;
    std::cout << "----------------------------" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i << ": " 
                  << layers[i].getInputSize() << " -> " 
                  << layers[i].getOutputSize() 
                  << " (" << layers[i].getActivationName() << ")" << std::endl;
    }
    std::cout << "Loss Function: " << lossFunction << std::endl;
    std::cout << "----------------------------" << std::endl;
}
