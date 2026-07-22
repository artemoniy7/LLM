#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>
#include <random>
#include <cmath>

class Tensor {
public:
    // Конструкторы
    Tensor() = default;
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, float value);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);
    
    // Доступ к данным
    float& at(const std::vector<int>& indices);
    float at(const std::vector<int>& indices) const;
    float& operator[](size_t idx);
    float operator[](size_t idx) const;
    
    // Свойства
    const std::vector<int>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    int dims() const { return shape_.size(); }
    
    // Инициализация
    void fill(float value);
    void random_uniform(float low = -1.0f, float high = 1.0f);
    void random_normal(float mean = 0.0f, float stddev = 1.0f);
    
    // Базовые операции
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    Tensor& operator+=(const Tensor& other);
    
    // Матричное умножение
    Tensor matmul(const Tensor& other) const;
    Tensor matmul_batch(const Tensor& other) const;
    
    // Транспонирование
    Tensor transpose() const;
    
    // Изменение формы
    Tensor reshape(const std::vector<int>& new_shape) const;
    
    // Вывод
    void print() const;
    
private:
    std::vector<int> shape_;
    std::vector<float> data_;
    
    int flat_index(const std::vector<int>& indices) const;
    bool shape_equal(const Tensor& other) const;
};

Tensor operator*(float scalar, const Tensor& tensor);