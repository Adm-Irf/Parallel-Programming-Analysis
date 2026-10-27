#include "matrix.hpp"
#include "parallel.hpp"
#include <string>
#include <iostream>
#include <stdexcept>
#include <iomanip>

// Constructors
Matrix::Matrix() : rows(0), cols(0), data(0) {}

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

Matrix::Matrix(size_t rows, size_t cols, double initValue) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, initValue)) {}

Matrix::Matrix(const std::vector<std::vector<double>>& data) : data(data), rows(data.size()), cols(data.empty() ? 0 : data[0].size()) {}

Matrix::Matrix(const Matrix& other) : data(other.data), rows(other.rows), cols(other.cols) {}

// Getters
size_t Matrix::getRows() const {
    return rows;
}

size_t Matrix::getCols() const {
    return cols;
}

double Matrix::getElement(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row][col];
}

const std::vector<std::vector<double>>& Matrix::getData() const {
    return data;
}

std::vector<std::vector<double>>& Matrix::getData() {
    return data;
}

// Setters
void Matrix::setElement(size_t row, size_t col, double value) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    data[row][col] = value;
}

void Matrix::setData(const std::vector<std::vector<double>>& newData) {
    data = newData;
    rows = newData.size();
    cols = newData.empty() ? 0 : newData[0].size();
}

Matrix Matrix::dot(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for dot product: (" + 
            std::to_string(rows) + "," + std::to_string(cols) + ") * (" + 
            std::to_string(other.rows) + "," + std::to_string(other.cols) + ")");
    }

    Matrix result(rows, other.cols, 0.0);
    
    // Optimized loop order: i-k-j for better cache locality
    // This accesses other.data[k][j] sequentially instead of randomly
// Each thread owns separate output rows, so threads never write the same value.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows > 1)
#endif
    for (int i = 0; i < static_cast<int>(rows); ++i) {
        for (size_t k = 0; k < cols; ++k) {
            double a_ik = data[i][k];
            if (a_ik == 0.0) continue;  // Skip zeros for sparse-friendly networks
            const double* other_row = other.data[k].data();
            double* result_row = result.data[i].data();
            for (size_t j = 0; j < other.cols; ++j) {
                result_row[j] += a_ik * other_row[j];
            }
        }
    }
    return result;
}

// Element-wise multiplication (Hadamard product)
Matrix Matrix::multiply(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication: (" + 
            std::to_string(rows) + "," + std::to_string(cols) + ") * (" + 
            std::to_string(other.rows) + "," + std::to_string(other.cols) + ")");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    return dot(other);
}

// Matrix addition
Matrix Matrix::add(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    return add(other);
}

// Matrix subtraction
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

// Scalar multiplication
Matrix Matrix::scalarMultiply(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    return scalarMultiply(scalar);
}

Matrix operator*(double scalar, const Matrix& m) {
    return m.scalarMultiply(scalar);
}

// Scalar addition to each element
Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] + scalar;
        }
    }
    return result;
}

// Scalar subtraction from each element
Matrix Matrix::operator-(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] - scalar;
        }
    }
    return result;
}

// Divide each element by scalar
Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] / scalar;
        }
    }
    return result;
}

// Transposition (MT)
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::T() const {
    return transpose();
}


Matrix Matrix::dotTranspose(const Matrix& left, const Matrix& right) const {
    size_t left_rows = left.getRows();
    size_t left_cols = left.getCols();
    size_t right_cols = right.getCols();
    
    if (left_rows != right.getRows()) {
        throw std::invalid_argument("dotTranspose: left rows must match right rows");
    }
    
    Matrix result(left_cols, right_cols, 0.0);
    const auto& left_data = left.getData();
    const auto& right_data = right.getData();
    
    // Compute left^T * right directly
    // result[j][b] = sum over a: left[a][j] * right[a][b]
// Each thread owns separate output rows, so threads never write the same value.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(left_rows > 1)
#endif
    for (int j = 0; j < static_cast<int>(left_cols); ++j) {
        for (size_t a = 0; a < left_rows; ++a) {
            double left_val = left_data[a][j];
            if (left_val == 0.0) continue;
            for (size_t b = 0; b < right_cols; ++b) {
                result.data[j][b] += left_val * right_data[a][b];
            }
        }
    }
    return result;
}

// Element-wise operations - apply a function to every element
Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = func(data[i][j]);
        }
    }
    return result;
}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        data = other.data;
        rows = other.rows;
        cols = other.cols;
    }
    return *this;
}

// Print matrix
void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << data[i][j] << " ";
        }
        std::cout << "]" << std::endl;
    }
}
