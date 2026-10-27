#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"
#include "activations.hpp"
#include <string>
#include <functional>

using ActivationFunction = std::function<Matrix(const Matrix&)>;
using ActivationDerivative = std::function<Matrix(const Matrix&)>;
using ActivationInPlace = std::function<void(Matrix&)>;

class Layer {
private:
    size_t inputSize;
    size_t outputSize;
    Matrix weights;
    Matrix biases;
    Matrix Z;
    Matrix A;
    Matrix prevA;
    Matrix dZ;  // Gradient of Z for backpropagation
    ActivationFunction activation;
    ActivationDerivative activationDerivative;
    ActivationInPlace activationInPlace;  // In-place variant for optimization
    std::string activationName;
    void initializeWeights();
    void initializeActivation(const std::string& name);

public:
    static void setSeed(unsigned int seed);
    Layer();
    Layer(size_t inputSize, size_t outputSize, const std::string& activationName = "relu");
    Layer(const Layer& other);
    ~Layer();

    Matrix forward(const Matrix& X);
    
    Matrix getWeights() const;
    Matrix getBiases() const;
    Matrix getZ() const;
    Matrix getA() const;
    Matrix getPrevA() const;
    Matrix getDZ() const;
    ActivationDerivative getActivationDerivative() const;
    ActivationInPlace getActivationInPlace() const;
    void setDZ(const Matrix& newDZ);
    size_t getInputSize() const;
    size_t getOutputSize() const;
    std::string getActivationName() const;
    
    void setWeights(const Matrix& newWeights);
    void setBiases(const Matrix& newBiases);
    void print() const;
    
    Layer& operator=(const Layer& other);
};

#endif
