#include "layer.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <istream>
#include <ostream>

// ============= Linear Layer =============
Linear::Linear(int in_features, int out_features)
    : weights_({out_features, in_features}),
      bias_({1, out_features}) {
    initialize_weights();
    grad_weights_ = Tensor({out_features, in_features}, 0.0f);
    grad_bias_ = Tensor({1, out_features}, 0.0f);

    weight_optimizer_ = std::make_shared<AdamOptimizer>(0.001f);
    bias_optimizer_ = std::make_shared<AdamOptimizer>(0.001f);
}

void Linear::initialize_weights() {
    float scale = std::sqrt(2.0f / weights_.shape()[1]);
    weights_.random_normal(0.0f, scale);
    bias_.fill(0.0f);
}

void Linear::set_optimizer(std::shared_ptr<AdamOptimizer> opt) {
    weight_optimizer_ = opt;
    bias_optimizer_ = std::make_shared<AdamOptimizer>(0.001f);
}

Tensor Linear::forward(const Tensor& input) {
    input_cache_ = input;

    if (input.dims() == 2) {
        Tensor output = input.matmul(weights_.transpose());

        for (int i = 0; i < output.shape()[0]; ++i) {
            for (int j = 0; j < output.shape()[1]; ++j) {
                output.at({i, j}) += bias_.at({0, j});
            }
        }
        return output;
    } else if (input.dims() == 3) {
        int batch_size = input.shape()[0];
        int seq_len = input.shape()[1];
        int in_features = input.shape()[2];

        if (in_features != weights_.shape()[1]) {
            throw std::runtime_error("Input feature dimension mismatch");
        }

        Tensor output({batch_size, seq_len, weights_.shape()[0]});

        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                Tensor token({1, in_features});
                for (int d = 0; d < in_features; ++d) {
                    token.at({0, d}) = input.at({b, s, d});
                }

                Tensor token_out = token.matmul(weights_.transpose());

                for (int j = 0; j < token_out.shape()[1]; ++j) {
                    token_out.at({0, j}) += bias_.at({0, j});
                }

                for (int j = 0; j < token_out.shape()[1]; ++j) {
                    output.at({b, s, j}) = token_out.at({0, j});
                }
            }
        }
        return output;
    } else {
        throw std::runtime_error("Linear layer supports only 2D or 3D tensors");
    }
}

Tensor Linear::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.dims() == 2) {
        Tensor grad_input = grad_output.matmul(weights_);

        Tensor input_t = input_cache_.transpose();
        grad_weights_ = input_t.matmul(grad_output);

        grad_bias_ = Tensor({1, grad_output.shape()[1]}, 0.0f);
        for (int i = 0; i < grad_output.shape()[0]; ++i) {
            for (int j = 0; j < grad_output.shape()[1]; ++j) {
                grad_bias_.at({0, j}) += grad_output.at({i, j});
            }
        }
    } else if (grad_output.dims() == 3) {
        int batch_size = grad_output.shape()[0];
        int seq_len = grad_output.shape()[1];
        int out_features = grad_output.shape()[2];
        int in_features = input_cache_.shape()[2];

        Tensor grad_input({batch_size, seq_len, in_features}, 0.0f);
        grad_weights_ = Tensor({out_features, in_features}, 0.0f);
        grad_bias_ = Tensor({1, out_features}, 0.0f);

        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                Tensor grad_token({1, out_features});
                for (int j = 0; j < out_features; ++j) {
                    grad_token.at({0, j}) = grad_output.at({b, s, j});
                }

                Tensor input_token({1, in_features});
                for (int d = 0; d < in_features; ++d) {
                    input_token.at({0, d}) = input_cache_.at({b, s, d});
                }

                Tensor grad_input_token = grad_token.matmul(weights_);
                for (int d = 0; d < in_features; ++d) {
                    grad_input.at({b, s, d}) = grad_input_token.at({0, d});
                }

                Tensor input_t = input_token.transpose();
                Tensor grad_weights_token = input_t.matmul(grad_token);
                for (int i = 0; i < out_features; ++i) {
                    for (int j = 0; j < in_features; ++j) {
                        grad_weights_.at({i, j}) += grad_weights_token.at({i, j});
                    }
                }

                for (int j = 0; j < out_features; ++j) {
                    grad_bias_.at({0, j}) += grad_output.at({b, s, j});
                }
            }
        }
    }

    update(learning_rate);
    return input_cache_;
}

void Linear::update(float learning_rate) {
    if (weight_optimizer_) {
        weight_optimizer_->update(weights_, grad_weights_);
        bias_optimizer_->update(bias_, grad_bias_);
    } else {
        for (size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] -= learning_rate * grad_weights_[i];
        }
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] -= learning_rate * grad_bias_[i];
        }
    }

    grad_weights_.fill(0.0f);
    grad_bias_.fill(0.0f);
}

std::vector<Tensor> Linear::parameters() const {
    return {weights_, bias_};
}

void Linear::save(std::ostream& out) const {
    weights_.save(out);
    bias_.save(out);
}

void Linear::load(std::istream& in) {
    Tensor weights;
    Tensor bias;
    weights.load(in);
    bias.load(in);

    if (weights.shape() != weights_.shape() || bias.shape() != bias_.shape()) {
        throw std::runtime_error("Linear checkpoint shape does not match model architecture");
    }

    weights_ = weights;
    bias_ = bias;
    grad_weights_ = Tensor(weights_.shape(), 0.0f);
    grad_bias_ = Tensor(bias_.shape(), 0.0f);
}

// ============= ReLU =============
Tensor ReLU::forward(const Tensor& input) {
    input_cache_ = input;
    Tensor output = input;
    for (size_t i = 0; i < output.size(); ++i) {
        if (output[i] < 0) output[i] = 0.0f;
    }
    return output;
}

Tensor ReLU::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input = grad_output;
    for (size_t i = 0; i < grad_input.size(); ++i) {
        if (input_cache_[i] <= 0) grad_input[i] = 0.0f;
    }
    return grad_input;
}

// ============= Tanh =============
Tensor Tanh::forward(const Tensor& input) {
    input_cache_ = input;
    Tensor output = input;
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
    return output;
}

Tensor Tanh::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input = grad_output;
    for (size_t i = 0; i < grad_input.size(); ++i) {
        float tanh_val = std::tanh(input_cache_[i]);
        grad_input[i] *= (1.0f - tanh_val * tanh_val);
    }
    return grad_input;
}

// ============= Cross-Entropy Loss (исправленный) =============
float CrossEntropyLoss::forward(const Tensor& predictions, const Tensor& targets) {
    predictions_cache_ = predictions;
    targets_cache_ = targets;

    // predictions: (batch_size, num_classes) или (1, num_classes)
    // targets: (batch_size, 1) или (1, 1)

    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];

    grad_cache_ = predictions;
    float total_loss = 0.0f;

    for (int i = 0; i < batch_size; ++i) {
        // Softmax для каждого элемента батча
        float max_val = predictions.at({i, 0});
        for (int j = 1; j < num_classes; ++j) {
            if (predictions.at({i, j}) > max_val) max_val = predictions.at({i, j});
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            grad_cache_.at({i, j}) = std::exp(predictions.at({i, j}) - max_val);
            sum_exp += grad_cache_.at({i, j});
        }

        for (int j = 0; j < num_classes; ++j) {
            grad_cache_.at({i, j}) /= sum_exp;
        }

        // Cross-Entropy
        int target_idx = (int)targets.at({i, 0});
        float log_prob = std::log(grad_cache_.at({i, target_idx}) + 1e-10f);
        total_loss -= log_prob;
    }

    return total_loss / batch_size;
}

Tensor CrossEntropyLoss::backward() {
    int batch_size = predictions_cache_.shape()[0];
    int num_classes = predictions_cache_.shape()[1];

    Tensor grad = grad_cache_;

    // Градиент: softmax - one_hot(target)
    for (int i = 0; i < batch_size; ++i) {
        int target_idx = (int)targets_cache_.at({i, 0});
        grad.at({i, target_idx}) -= 1.0f;
    }

    // Усредняем по батчу
    Tensor result(grad.shape());
    for (size_t i = 0; i < grad.size(); ++i) {
        result[i] = grad[i] / batch_size;
    }

    return result;
}

// ============= Sequential Model =============
void Sequential::add_layer(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;
    for (auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}

void Sequential::backward(const Tensor& grad, float learning_rate) {
    Tensor grad_output = grad;
    for (int i = layers_.size() - 1; i >= 0; --i) {
        grad_output = layers_[i]->backward(grad_output, learning_rate);
    }
}

void Sequential::update(float learning_rate) {
    for (auto& layer : layers_) {
        layer->update(learning_rate);
    }
}

void Sequential::train(const Tensor& input, const Tensor& target, float learning_rate) {
    learning_rate_ = learning_rate;
    Tensor output = forward(input);
    CrossEntropyLoss loss_fn;
    float loss = loss_fn.forward(output, target);

    Tensor grad = loss_fn.backward();
    backward(grad, learning_rate);
    update(learning_rate);

    static int step = 0;
    if (step % 100 == 0) {
        std::cout << "Step " << step << ", Loss: " << loss << std::endl;
    }
    step++;
}

void Sequential::print_parameters() const {
    std::cout << "Model parameters:\n";
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto params = layers_[i]->parameters();
        if (!params.empty()) {
            std::cout << "Layer " << i << ": ";
            for (const auto& p : params) {
                std::cout << p.shape()[0] << "x" << p.shape()[1] << " ";
            }
            std::cout << std::endl;
        }
    }
}