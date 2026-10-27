#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "matrix.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

/**
 * Activations namespace for neural network activation functions
 * and their derivatives used in backpropagation.
 */
namespace Activations {
    
    /**
     * Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
     * Maps values to (0, 1) range - useful for hidden layers
     * and binary classification output.
     */
    Matrix sigmoid(const Matrix& x);
    
    /**
     * Derivative of sigmoid: d/dx σ(x) = σ(x) * (1 - σ(x))
     * Used in backpropagation to compute gradients.
     * @param x The sigmoid-activated values (not the raw input!)
     */
    Matrix sigmoid_derivative(const Matrix& x);
    
    /**
     * ReLU (Rectified Linear Unit) activation: max(0, x)
     * Introduces non-linearity while being computationally efficient.
     * Common choice for hidden layers in deep networks.
     */
    Matrix relu(const Matrix& x);
    
    /**
     * Derivative of ReLU: 1 if x > 0, 0 otherwise
     * @param x The ReLU-activated values (not the raw input!)
     */
    Matrix relu_derivative(const Matrix& x);
    
    /**
     * Leaky ReLU: x > 0 ? x : 0.01 * x
     * Addresses the "dying ReLU" problem by allowing small gradient when x < 0.
     */
    Matrix leaky_relu(const Matrix& x, double alpha = 0.01);
    
    /**
     * Derivative of Leaky ReLU: 1 if x > 0, alpha otherwise
     */
    Matrix leaky_relu_derivative(const Matrix& x, double alpha = 0.01);
    
    /**
     * Softmax activation: e^x_i / Σ(e^x_j)
     * Converts raw scores into probability distribution summing to 1.
     * Typically used for multi-class classification output (e.g., digits 0-9).
     * Applied along rows (each row is a sample).
     */
    Matrix softmax(const Matrix& x);
    
    /**
     * Derivative of softmax with respect to input.
     * More complex than other activations - depends on Jacobian matrix.
     * For backpropagation, we use: dL/dx = softmax(x) * (dL/dy - Σ(dL/dy * softmax(x)))
     * @param x The softmax-activated values (probabilities)
     * @param upstream_grad The gradient from the layer above
     */
    Matrix softmax_derivative(const Matrix& x, const Matrix& upstream_grad);
    
    /**
     * Tanh activation: tanh(x)
     * Maps values to (-1, 1) range. Zero-centered.
     */
    Matrix tanh(const Matrix& x);
    
    /**
     * Derivative of tanh: d/dx tanh(x) = 1 - tanh²(x)
     * @param x The tanh-activated values
     */
    Matrix tanh_derivative(const Matrix& x);
    
    /**
     * Element-wise identity/linear activation: f(x) = x
     * Useful for output layers in regression.
     */
    Matrix linear(const Matrix& x);
    
    /**
     * Derivative of linear: 1
     */
    Matrix linear_derivative(const Matrix& x);
    
    // In-place variants (optimized - avoid memory allocation)
    void sigmoid_inplace(Matrix& x);
    void relu_inplace(Matrix& x);
    void leaky_relu_inplace(Matrix& x, double alpha = 0.01);
    void tanh_inplace(Matrix& x);
    void linear_inplace(Matrix& x);

} // namespace Activations

#endif // ACTIVATIONS_HPP
