#include "activations.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <iostream>

namespace Activations {
    
    Matrix sigmoid(const Matrix& x) {
        return x.apply([](double val) {
            return 1.0 / (1.0 + std::exp(-val));
        });
    }
    
    Matrix sigmoid_derivative(const Matrix& x) {
        // Derivative of sigmoid: σ(x) * (1 - σ(x))
        // Since we receive Z (pre-activation), we need to compute sigmoid(Z) first
        return x.apply([](double val) {
            double sigmoid_val = 1.0 / (1.0 + std::exp(-val));
            return sigmoid_val * (1.0 - sigmoid_val);
        });
    }
    
    // ReLU: max(0, x)
    Matrix relu(const Matrix& x) {
        return x.apply([](double val) {
            return std::max(0.0, val);
        });
    }
    
    // ReLU derivative: 1 if x > 0, 0 otherwise
    Matrix relu_derivative(const Matrix& x) {
        return x.apply([](double val) {
            return val > 0.0 ? 1.0 : 0.0;
        });
    }
    
    // Leaky ReLU: x > 0 ? x : alpha * x
    Matrix leaky_relu(const Matrix& x, double alpha) {
        return x.apply([alpha](double val) {
            return val > 0.0 ? val : alpha * val;
        });
    }
    
    // Leaky ReLU derivative: 1 if x > 0, alpha otherwise
    Matrix leaky_relu_derivative(const Matrix& x, double alpha) {
        return x.apply([alpha](double val) {
            return val > 0.0 ? 1.0 : alpha;
        });
    }
    
    // Softmax: e^x_i / Σ(e^x_j) for each row
    // Applied along rows - each row is treated as an independent sample
    Matrix softmax(const Matrix& x) {
        size_t rows = x.getRows();
        size_t cols = x.getCols();
        
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be non-zero for softmax");
        }
        
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
        const auto& data = x.getData();
        
        for (size_t i = 0; i < rows; ++i) {
            // Find max for numerical stability (subtract before exp)
            double max_val = data[i][0];
            for (size_t j = 1; j < cols; ++j) {
                if (data[i][j] > max_val) {
                    max_val = data[i][j];
                }
            }
            
            // Compute exponentials
            double sum_exp = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = std::exp(data[i][j] - max_val);
                sum_exp += result[i][j];
            }
            
            // Normalize
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] /= sum_exp;
            }
        }
        
        return Matrix(result);
    }
    
    
    Matrix softmax_derivative(const Matrix& x, const Matrix& upstream_grad) {
        size_t rows = x.getRows();
        size_t cols = x.getCols();
        
        if (rows != upstream_grad.getRows() || cols != upstream_grad.getCols()) {
            throw std::invalid_argument("Softmax derivative: dimensions must match between output and upstream gradient");
        }
        
        const auto& x_data = x.getData();
        const auto& grad_data = upstream_grad.getData();
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
        
        for (size_t i = 0; i < rows; ++i) {
            // Compute Σ(dL/dy * softmax(x)) for this row
            double dot_product = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                dot_product += grad_data[i][j] * x_data[i][j];
            }
            
            // Compute dL/dx = softmax(x) * (dL/dy - Σ)
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = x_data[i][j] * (grad_data[i][j] - dot_product);
            }
        }
        
        return Matrix(result);
    }
    
    // Tanh: tanh(x)
    Matrix tanh(const Matrix& x) {
        return x.apply([](double val) {
            return std::tanh(val);
        });
    }
    
    // Tanh derivative: 1 - tanh²(x)
    // Since we receive Z (pre-activation), we need to compute tanh(Z) first
    Matrix tanh_derivative(const Matrix& x) {
        return x.apply([](double val) {
            double tanh_val = std::tanh(val);
            return 1.0 - tanh_val * tanh_val;
        });
    }
    
    // Linear: f(x) = x
    Matrix linear(const Matrix& x) {
        return Matrix(x);  // Return a copy
    }
    
    // Linear derivative: 1
    Matrix linear_derivative(const Matrix& x) {
        return x.apply([](double val) {
            (void)val;  // Suppress unused parameter warning
            return 1.0;
        });
    }
    
    // In-place sigmoid (optimized)
    void sigmoid_inplace(Matrix& x) {
        auto& data = x.getData();
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                data[i][j] = 1.0 / (1.0 + std::exp(-data[i][j]));
            }
        }
    }
    
    // In-place ReLU (optimized)
    void relu_inplace(Matrix& x) {
        auto& data = x.getData();
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                data[i][j] = std::max(0.0, data[i][j]);
            }
        }
    }
    
    // In-place Leaky ReLU (optimized)
    void leaky_relu_inplace(Matrix& x, double alpha) {
        auto& data = x.getData();
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                data[i][j] = data[i][j] > 0.0 ? data[i][j] : alpha * data[i][j];
            }
        }
    }
    
    // In-place tanh (optimized)
    void tanh_inplace(Matrix& x) {
        auto& data = x.getData();
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                data[i][j] = std::tanh(data[i][j]);
            }
        }
    }
    
    // In-place linear (no-op, just for API consistency)
    void linear_inplace(Matrix& x) {
        (void)x;  // Linear is identity, no change needed
    }
    
} // namespace Activations
