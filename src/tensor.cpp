#include "tensor.hpp"
#include <numeric>
#include <cmath>
#include <cassert>

// ---------- Конструкторы ----------
Tensor::Tensor(const std::vector<int>& shape) 
    : shape_(shape) {
    size_t total = 1;
    for (int dim : shape_) total *= dim;
    data_.resize(total, 0.0f);
}

Tensor::Tensor(const std::vector<int>& shape, float value) 
    : shape_(shape) {
    size_t total = 1;
    for (int dim : shape_) total *= dim;
    data_.assign(total, value);
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) 
    : shape_(shape), data_(data) {
    size_t total = 1;
    for (int dim : shape_) total *= dim;
    if (data_.size() != total) {
        throw std::runtime_error("Data size does not match shape");
    }
}

// ---------- Доступ к данным ----------
int Tensor::flat_index(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Index dimension mismatch");
    }
    int idx = 0;
    int stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds");
        }
        idx += indices[i] * stride;
        stride *= shape_[i];
    }
    return idx;
}

float& Tensor::at(const std::vector<int>& indices) {
    return data_[flat_index(indices)];
}

float Tensor::at(const std::vector<int>& indices) const {
    return data_[flat_index(indices)];
}

float& Tensor::operator[](size_t idx) {
    return data_[idx];
}

float Tensor::operator[](size_t idx) const {
    return data_[idx];
}

// ---------- Инициализация ----------
void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::random_uniform(float low, float high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& val : data_) val = dist(gen);
}

void Tensor::random_normal(float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    for (auto& val : data_) val = dist(gen);
}

// ---------- Операции ----------
bool Tensor::shape_equal(const Tensor& other) const {
    if (shape_.size() != other.shape_.size()) return false;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != other.shape_[i]) return false;
    }
    return true;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (!shape_equal(other)) {
        throw std::runtime_error("Shape mismatch in addition");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (!shape_equal(other)) {
        throw std::runtime_error("Shape mismatch in subtraction");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (!shape_equal(other)) {
        throw std::runtime_error("Shape mismatch in element-wise multiplication");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (!shape_equal(other)) {
        throw std::runtime_error("Shape mismatch in addition");
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

// ---------- Матричное умножение (2D) ----------
Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("matmul supports only 2D tensors");
    }
    
    int rows = shape_[0];
    int cols = shape_[1];
    int other_rows = other.shape_[0];
    int other_cols = other.shape_[1];
    
    if (cols != other_rows) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }
    
    Tensor result({rows, other_cols}, 0.0f);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols; ++k) {
                sum += data_[i * cols + k] * other.data_[k * other_cols + j];
            }
            result.data_[i * other_cols + j] = sum;
        }
    }
    return result;
}

// ---------- Матричное умножение с батчем (3D) ----------
Tensor Tensor::matmul_batch(const Tensor& other) const {
    if (shape_.size() != 3) {
        throw std::runtime_error("matmul_batch requires 3D tensor as first argument");
    }
    
    int batch_size = shape_[0];
    int seq_len = shape_[1];
    int d_model = shape_[2];
    
    int other_rows, other_cols;
    
    if (other.shape_.size() == 2) {
        other_rows = other.shape_[0];
        other_cols = other.shape_[1];
    } else if (other.shape_.size() == 3) {
        if (other.shape_[0] != batch_size) {
            throw std::runtime_error("Batch size mismatch in matmul_batch");
        }
        other_rows = other.shape_[2];
        other_cols = other.shape_[2];
    } else {
        throw std::runtime_error("matmul_batch requires 2D or 3D tensor as second argument");
    }
    
    if (d_model != other_rows) {
        throw std::runtime_error("Dimension mismatch in matmul_batch");
    }
    
    Tensor result({batch_size, seq_len, other_cols}, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < other_cols; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < d_model; ++k) {
                    float a_val = at({b, i, k});
                    float b_val;
                    if (other.shape_.size() == 2) {
                        b_val = other.at({k, j});
                    } else {
                        b_val = other.at({b, k, j});
                    }
                    sum += a_val * b_val;
                }
                result.at({b, i, j}) = sum;
            }
        }
    }
    
    return result;
}

// ---------- Транспонирование ----------
Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Transpose supports only 2D tensors");
    }
    std::vector<int> new_shape = {shape_[1], shape_[0]};
    Tensor result(new_shape);
    
    for (int i = 0; i < shape_[0]; ++i) {
        for (int j = 0; j < shape_[1]; ++j) {
            result.data_[j * shape_[0] + i] = data_[i * shape_[1] + j];
        }
    }
    return result;
}

// ---------- Изменение формы ----------
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    size_t new_size = 1;
    for (int dim : new_shape) new_size *= dim;
    if (new_size != data_.size()) {
        throw std::runtime_error("New shape size doesn't match tensor size");
    }
    Tensor result(new_shape);
    result.data_ = data_;
    return result;
}

// ---------- Вывод ----------
void Tensor::print() const {
    std::cout << "Tensor shape: (";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << ")\nData: [";
    for (size_t i = 0; i < std::min(data_.size(), size_t(10)); ++i) {
        std::cout << data_[i];
        if (i < std::min(data_.size(), size_t(10)) - 1) std::cout << ", ";
    }
    if (data_.size() > 10) std::cout << ", ...";
    std::cout << "]\n";
}