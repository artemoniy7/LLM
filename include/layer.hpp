#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>
#include <cmath>

// Базовый класс для всех слоев
class Layer {
public:
    virtual ~Layer() = default;
    
    // Прямой проход (forward pass)
    virtual Tensor forward(const Tensor& input) = 0;
    
    // Обратный проход (backward pass)
    virtual Tensor backward(const Tensor& grad_output, float learning_rate) = 0;
    
    // Обновление параметров
    virtual void update(float learning_rate) = 0;
    
    // Получение параметров для сохранения
    virtual std::vector<Tensor> parameters() const { return {}; }
};

// ============= Adam Optimizer =============
class AdamOptimizer {
public:
    AdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {}
    
    void update(Tensor& param, const Tensor& grad) {
        t_++;
        
        // Инициализируем моменты, если нужно
        if (m_.size() != param.size()) {
            m_ = Tensor(param.shape(), 0.0f);
            v_ = Tensor(param.shape(), 0.0f);
        }
        
        // Обновляем моменты
        float beta1_t = 1.0f - std::pow(beta1_, t_);
        float beta2_t = 1.0f - std::pow(beta2_, t_);
        
        for (size_t i = 0; i < param.size(); ++i) {
            m_[i] = beta1_ * m_[i] + (1.0f - beta1_) * grad[i];
            v_[i] = beta2_ * v_[i] + (1.0f - beta2_) * grad[i] * grad[i];
            
            // Обновляем параметры
            param[i] -= lr_ * (m_[i] / beta1_t) / (std::sqrt(v_[i] / beta2_t) + eps_);
        }
    }
    
private:
    float lr_, beta1_, beta2_, eps_;
    int t_;
    Tensor m_, v_;
};

// ============= Linear Layer =============
class Linear : public Layer {
public:
    Linear(int in_features, int out_features);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void update(float learning_rate) override;
    
    std::vector<Tensor> parameters() const override;
    void set_optimizer(std::shared_ptr<AdamOptimizer> opt);
    
private:
    Tensor weights_;
    Tensor bias_;
    Tensor input_cache_;  // для backward pass
    
    // Градиенты
    Tensor grad_weights_;
    Tensor grad_bias_;
    
    std::shared_ptr<AdamOptimizer> weight_optimizer_;
    std::shared_ptr<AdamOptimizer> bias_optimizer_;
    
    void initialize_weights();
};

// ============= ReLU Activation =============
class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void update(float learning_rate) override {}
    
private:
    Tensor input_cache_;
};

// ============= Tanh Activation =============
class Tanh : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void update(float learning_rate) override {}
    
private:
    Tensor input_cache_;
};

// ============= Cross-Entropy Loss =============
class CrossEntropyLoss {
public:
    float forward(const Tensor& predictions, const Tensor& targets);
    Tensor backward();
    
private:
    Tensor predictions_cache_;
    Tensor targets_cache_;
    Tensor grad_cache_;
};

// ============= Sequential Model =============
class Sequential {
public:
    void add_layer(std::shared_ptr<Layer> layer);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad, float learning_rate);
    void update(float learning_rate);
    void train(const Tensor& input, const Tensor& target, float learning_rate);
    
    // Для отладки
    void print_parameters() const;
    void set_learning_rate(float lr) { learning_rate_ = lr; }
    
private:
    std::vector<std::shared_ptr<Layer>> layers_;
    float learning_rate_ = 0.01f;
};