#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <functional>
#include <cstddef>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double initValue);
    Matrix(const std::vector<std::vector<double>>& data);
    Matrix(const Matrix& other);

    size_t getRows() const;
    size_t getCols() const;
    double getElement(size_t row, size_t col) const;
    const std::vector<std::vector<double>>& getData() const;
    std::vector<std::vector<double>>& getData();

    void setElement(size_t row, size_t col, double value);
    void setData(const std::vector<std::vector<double>>& newData);

    // Matrix operations
    Matrix dot(const Matrix& other) const;                    // Matrix multiplication (dot product)
    Matrix multiply(const Matrix& other) const;                // Element-wise multiplication (Hadamard product)
    Matrix operator*(const Matrix& other) const;               // Matrix multiplication operator
    Matrix add(const Matrix& other) const;                     // Matrix addition
    Matrix operator+(const Matrix& other) const;               // Matrix addition operator
    Matrix operator-(const Matrix& other) const;               // Matrix subtraction operator
    Matrix scalarMultiply(double scalar) const;               // Scalar multiplication
    Matrix operator*(double scalar) const;                     // Scalar multiplication operator
    friend Matrix operator*(double scalar, const Matrix& m);   // Friend function for scalar * matrix
    Matrix transpose() const;                                  // Transposition (MT)
    Matrix T() const;                                          // Transposition (alias)
    
    // Optimized operations
    Matrix dotTranspose(const Matrix& left, const Matrix& right) const;  // Compute left^T * right without explicit transpose

    Matrix apply(std::function<double(double)> func) const;    // Apply function to each element
    Matrix operator+(double scalar) const;                      // Add scalar to each element
    Matrix operator-(double scalar) const;                      // Subtract scalar from each element
    Matrix operator/(double scalar) const;                       // Divide each element by scalar

    void print() const;
    Matrix& operator=(const Matrix& other);
};

#endif 
