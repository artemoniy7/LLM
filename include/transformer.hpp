#pragma once

#include "tensor.hpp"
#include "layer.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

// ============= Multi-Head Attention =============
class MultiHeadAttention : public Layer {
public:
    MultiHeadAttention(int d_model, int num_heads);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void update(float learning_rate) override;
    
    std::vector<Tensor> parameters() const override;
    
private:
    int d_model_;
    int num_heads_;
    int d_k_;
    float scale_;
    
    std::shared_ptr<Linear> w_q_;
    std::shared_ptr<Linear> w_k_;
    std::shared_ptr<Linear> w_v_;
    std::shared_ptr<Linear> w_o_;
    
    Tensor q_cache_;
    Tensor k_cache_;
    Tensor v_cache_;
    Tensor scores_cache_;
    Tensor attention_cache_;
    
    Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v);
};

// ============= Feed-Forward Network =============
class FeedForward : public Layer {
public:
    FeedForward(int d_model, int d_ff);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void update(float learning_rate) override;
    
    std::vector<Tensor> parameters() const override;
    
private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    Tensor input_cache_;
};

// ============= Layer Normalization =============
class LayerNorm : public Layer {
public:
    LayerNorm(int d_model, float eps = 1e-5f);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void update(float learning_rate) override;
    
    std::vector<Tensor> parameters() const override;
    
private:
    int d_model_;
    float eps_;
    Tensor gamma_;
    Tensor beta_;
    Tensor input_cache_;
    Tensor mean_cache_;
    Tensor var_cache_;
};

// ============= Transformer Block =============
class TransformerBlock : public Layer {
public:
    TransformerBlock(int d_model, int num_heads, int d_ff);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void update(float learning_rate) override;
    
    std::vector<Tensor> parameters() const override;
    
private:
    std::shared_ptr<MultiHeadAttention> attention_;
    std::shared_ptr<LayerNorm> norm1_;
    std::shared_ptr<LayerNorm> norm2_;
    std::shared_ptr<FeedForward> ff_;
    
    Tensor attn_cache_;
    Tensor ff_cache_;
};

// ============= Positional Encoding =============
class PositionalEncoding {
public:
    PositionalEncoding(int d_model, int max_len = 5000);
    Tensor forward(const Tensor& input);
    
private:
    Tensor pe_;
};

// ============= GPT Model =============
class GPT : public Layer {
public:
    GPT(int vocab_size, int d_model, int num_heads, int num_layers, int d_ff, int max_len = 1024);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void update(float learning_rate) override;
    
    std::vector<Tensor> parameters() const override;
    
    // Публичные методы для генерации и получения логитов
    Tensor get_logits(const Tensor& hidden_states);
    std::vector<int> generate(const std::vector<int>& prompt, int max_tokens, float temperature = 1.0f);
    
private:
    int vocab_size_;
    int d_model_;
    int max_len_;
    
    Tensor token_embeddings_;
    PositionalEncoding pos_encoding_;
    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
    std::shared_ptr<LayerNorm> final_norm_;
    std::shared_ptr<Linear> lm_head_;
    
    Tensor input_cache_;
    
    int sample_token(const Tensor& logits, float temperature);
};