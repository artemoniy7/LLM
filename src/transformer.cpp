#include "transformer.hpp"
#include <algorithm>
#include <random>

// ============= Multi-Head Attention =============
MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model_(d_model), num_heads_(num_heads) {
    
    d_k_ = d_model / num_heads;
    scale_ = 1.0f / std::sqrt(d_k_);
    
    w_q_ = std::make_shared<Linear>(d_model, d_model);
    w_k_ = std::make_shared<Linear>(d_model, d_model);
    w_v_ = std::make_shared<Linear>(d_model, d_model);
    w_o_ = std::make_shared<Linear>(d_model, d_model);
}

Tensor MultiHeadAttention::scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
    int batch_size = q.shape()[0];
    int seq_len = q.shape()[1];
    
    // Изменяем форму для многоголового внимания
    Tensor q_reshaped({batch_size, num_heads_, seq_len, d_k_});
    Tensor k_reshaped({batch_size, num_heads_, seq_len, d_k_});
    Tensor v_reshaped({batch_size, num_heads_, seq_len, d_k_});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < d_k_; ++d) {
                    q_reshaped.at({b, h, s, d}) = q.at({b, s, h * d_k_ + d});
                    k_reshaped.at({b, h, s, d}) = k.at({b, s, h * d_k_ + d});
                    v_reshaped.at({b, h, s, d}) = v.at({b, s, h * d_k_ + d});
                }
            }
        }
    }
    
    // Вычисляем scores
    Tensor scores({batch_size, num_heads_, seq_len, seq_len});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < d_k_; ++d) {
                        dot += q_reshaped.at({b, h, i, d}) * k_reshaped.at({b, h, j, d});
                    }
                    scores.at({b, h, i, j}) = dot * scale_;
                }
            }
        }
    }
    
    // === CAUSAL MASK ===
    // Запрещаем модели "видеть" будущие токены
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = i + 1; j < seq_len; ++j) {
                    scores.at({b, h, i, j}) = -1e9f;  // очень маленькое число
                }
            }
        }
    }
    
    // Softmax
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                float max_val = scores.at({b, h, i, 0});
                for (int j = 1; j < seq_len; ++j) {
                    if (scores.at({b, h, i, j}) > max_val) max_val = scores.at({b, h, i, j});
                }
                
                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    scores.at({b, h, i, j}) = std::exp(scores.at({b, h, i, j}) - max_val);
                    sum_exp += scores.at({b, h, i, j});
                }
                
                for (int j = 0; j < seq_len; ++j) {
                    scores.at({b, h, i, j}) /= sum_exp;
                }
            }
        }
    }
    
    scores_cache_ = scores;
    
    // attention = scores @ v
    Tensor attention({batch_size, num_heads_, seq_len, d_k_});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                for (int d = 0; d < d_k_; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        sum += scores.at({b, h, i, j}) * v_reshaped.at({b, h, j, d});
                    }
                    attention.at({b, h, i, d}) = sum;
                }
            }
        }
    }
    
    // Объединяем heads
    Tensor combined({batch_size, seq_len, d_model_});
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads_; ++h) {
                for (int d = 0; d < d_k_; ++d) {
                    combined.at({b, s, h * d_k_ + d}) = attention.at({b, h, s, d});
                }
            }
        }
    }
    
    attention_cache_ = combined;
    return combined;
}

Tensor MultiHeadAttention::forward(const Tensor& input) {
    // input: (batch_size, seq_len, d_model)
    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];
    
    // Применяем линейные слои к каждому токену отдельно
    Tensor q({batch_size, seq_len, d_model_});
    Tensor k({batch_size, seq_len, d_model_});
    Tensor v({batch_size, seq_len, d_model_});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            // Берем один токен
            Tensor token({1, d_model_});
            for (int d = 0; d < d_model_; ++d) {
                token.at({0, d}) = input.at({b, s, d});
            }
            
            // Применяем линейные слои
            Tensor q_token = w_q_->forward(token);
            Tensor k_token = w_k_->forward(token);
            Tensor v_token = w_v_->forward(token);
            
            // Сохраняем результаты
            for (int d = 0; d < d_model_; ++d) {
                q.at({b, s, d}) = q_token.at({0, d});
                k.at({b, s, d}) = k_token.at({0, d});
                v.at({b, s, d}) = v_token.at({0, d});
            }
        }
    }
    
    q_cache_ = q;
    k_cache_ = k;
    v_cache_ = v;
    
    // Scaled dot-product attention
    Tensor attn = scaled_dot_product_attention(q, k, v);
    
    // Выходная проекция (применяем к каждому токену)
    Tensor output({batch_size, seq_len, d_model_});
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            Tensor token({1, d_model_});
            for (int d = 0; d < d_model_; ++d) {
                token.at({0, d}) = attn.at({b, s, d});
            }
            
            Tensor out_token = w_o_->forward(token);
            for (int d = 0; d < d_model_; ++d) {
                output.at({b, s, d}) = out_token.at({0, d});
            }
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad = w_o_->backward(grad_output, learning_rate);
    return grad;
}

void MultiHeadAttention::update(float learning_rate) {
    w_q_->update(learning_rate);
    w_k_->update(learning_rate);
    w_v_->update(learning_rate);
    w_o_->update(learning_rate);
}

std::vector<Tensor> MultiHeadAttention::parameters() const {
    auto params = w_q_->parameters();
    auto p2 = w_k_->parameters();
    auto p3 = w_v_->parameters();
    auto p4 = w_o_->parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    params.insert(params.end(), p4.begin(), p4.end());
    return params;
}

// ============= Feed-Forward Network =============
FeedForward::FeedForward(int d_model, int d_ff) {
    fc1_ = std::make_shared<Linear>(d_model, d_ff);
    fc2_ = std::make_shared<Linear>(d_ff, d_model);
}

Tensor FeedForward::forward(const Tensor& input) {
    input_cache_ = input;
    Tensor hidden = fc1_->forward(input);
    
    for (size_t i = 0; i < hidden.size(); ++i) {
        if (hidden[i] < 0) hidden[i] = 0.0f;
    }
    
    return fc2_->forward(hidden);
}

Tensor FeedForward::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad = fc2_->backward(grad_output, learning_rate);
    
    for (size_t i = 0; i < grad.size(); ++i) {
        if (grad[i] < 0) grad[i] = 0.0f;
    }
    
    return fc1_->backward(grad, learning_rate);
}

void FeedForward::update(float learning_rate) {
    fc1_->update(learning_rate);
    fc2_->update(learning_rate);
}

std::vector<Tensor> FeedForward::parameters() const {
    auto params = fc1_->parameters();
    auto p2 = fc2_->parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    return params;
}

// ============= Layer Normalization =============
LayerNorm::LayerNorm(int d_model, float eps) 
    : d_model_(d_model), eps_(eps) {
    gamma_ = Tensor({1, d_model}, 1.0f);
    beta_ = Tensor({1, d_model}, 0.0f);
}

Tensor LayerNorm::forward(const Tensor& input) {
    input_cache_ = input;
    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];
    
    Tensor output({batch_size, seq_len, d_model_});
    mean_cache_ = Tensor({batch_size, seq_len});
    var_cache_ = Tensor({batch_size, seq_len});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            float mean = 0.0f;
            for (int d = 0; d < d_model_; ++d) {
                mean += input.at({b, s, d});
            }
            mean /= d_model_;
            mean_cache_.at({b, s}) = mean;
            
            float var = 0.0f;
            for (int d = 0; d < d_model_; ++d) {
                float diff = input.at({b, s, d}) - mean;
                var += diff * diff;
            }
            var /= d_model_;
            var_cache_.at({b, s}) = var;
            
            float inv_std = 1.0f / std::sqrt(var + eps_);
            for (int d = 0; d < d_model_; ++d) {
                float normalized = (input.at({b, s, d}) - mean) * inv_std;
                output.at({b, s, d}) = gamma_.at({0, d}) * normalized + beta_.at({0, d});
            }
        }
    }
    
    return output;
}

Tensor LayerNorm::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input = grad_output;
    return grad_input;
}

void LayerNorm::update(float learning_rate) {
    for (size_t i = 0; i < gamma_.size(); ++i) {
        gamma_[i] -= learning_rate * 0.001f;
        beta_[i] -= learning_rate * 0.001f;
    }
}

std::vector<Tensor> LayerNorm::parameters() const {
    return {gamma_, beta_};
}

// ============= Transformer Block =============
TransformerBlock::TransformerBlock(int d_model, int num_heads, int d_ff) {
    attention_ = std::make_shared<MultiHeadAttention>(d_model, num_heads);
    norm1_ = std::make_shared<LayerNorm>(d_model);
    norm2_ = std::make_shared<LayerNorm>(d_model);
    ff_ = std::make_shared<FeedForward>(d_model, d_ff);
}

Tensor TransformerBlock::forward(const Tensor& input) {
    // Self-attention with residual
    Tensor attn_out = attention_->forward(input);
    attn_cache_ = attn_out;
    Tensor x = attn_out + input;
    x = norm1_->forward(x);
    
    // Feed-forward with residual
    Tensor ff_out = ff_->forward(x);
    ff_cache_ = ff_out;
    x = ff_out + x;
    x = norm2_->forward(x);
    
    return x;
}

Tensor TransformerBlock::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad = grad_output;
    grad = norm2_->backward(grad, learning_rate);
    grad = ff_->backward(grad, learning_rate);
    grad = grad + ff_cache_;
    
    grad = norm1_->backward(grad, learning_rate);
    grad = attention_->backward(grad, learning_rate);
    grad = grad + attn_cache_;
    
    return grad;
}

void TransformerBlock::update(float learning_rate) {
    attention_->update(learning_rate);
    norm1_->update(learning_rate);
    ff_->update(learning_rate);
    norm2_->update(learning_rate);
}

std::vector<Tensor> TransformerBlock::parameters() const {
    auto params = attention_->parameters();
    auto p2 = norm1_->parameters();
    auto p3 = ff_->parameters();
    auto p4 = norm2_->parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    params.insert(params.end(), p4.begin(), p4.end());
    return params;
}

// ============= Positional Encoding =============
PositionalEncoding::PositionalEncoding(int d_model, int max_len) {
    pe_ = Tensor({max_len, d_model});
    
    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            float angle = pos / std::pow(10000.0f, (2.0f * i) / d_model);
            if (i % 2 == 0) {
                pe_.at({pos, i}) = std::sin(angle);
            } else {
                pe_.at({pos, i}) = std::cos(angle);
            }
        }
    }
}

Tensor PositionalEncoding::forward(const Tensor& input) {
    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];
    int d_model = input.shape()[2];
    
    Tensor output = input;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int d = 0; d < d_model; ++d) {
                output.at({b, s, d}) += pe_.at({s, d});
            }
        }
    }
    
    return output;
}

// ============= GPT Model =============
GPT::GPT(int vocab_size, int d_model, int num_heads, int num_layers, int d_ff, int max_len)
    : vocab_size_(vocab_size), d_model_(d_model), max_len_(max_len),
      pos_encoding_(d_model, max_len) {
    
    token_embeddings_ = Tensor({vocab_size, d_model});
    token_embeddings_.random_normal(0.0f, 0.02f);
    
    for (int i = 0; i < num_layers; ++i) {
        blocks_.push_back(std::make_shared<TransformerBlock>(d_model, num_heads, d_ff));
    }
    
    final_norm_ = std::make_shared<LayerNorm>(d_model);
    lm_head_ = std::make_shared<Linear>(d_model, vocab_size);
}

Tensor GPT::forward(const Tensor& input) {
    input_cache_ = input;
    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];
    
    // Embedding: (batch_size, seq_len, d_model)
    Tensor hidden({batch_size, seq_len, d_model_});
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int token_id = (int)input.at({b, s});
            for (int d = 0; d < d_model_; ++d) {
                hidden.at({b, s, d}) = token_embeddings_.at({token_id, d});
            }
        }
    }
    
    // Positional encoding
    hidden = pos_encoding_.forward(hidden);
    
    // Transformer blocks
    for (auto& block : blocks_) {
        hidden = block->forward(hidden);
    }
    
    // Final layer norm
    hidden = final_norm_->forward(hidden);
    
    return hidden;
}

Tensor GPT::get_logits(const Tensor& hidden_states) {
    int batch_size = hidden_states.shape()[0];
    int seq_len = hidden_states.shape()[1];
    
    Tensor logits({batch_size, seq_len, vocab_size_});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            Tensor token_emb({1, d_model_});
            for (int d = 0; d < d_model_; ++d) {
                token_emb.at({0, d}) = hidden_states.at({b, s, d});
            }
            
            Tensor out = lm_head_->forward(token_emb);
            for (int v = 0; v < vocab_size_; ++v) {
                logits.at({b, s, v}) = out.at({0, v});
            }
        }
    }
    
    return logits;
}

int GPT::sample_token(const Tensor& logits, float temperature) {
    int vocab_size = logits.shape()[1];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits.at({0, i}) / temperature);
        sum += probs[i];
    }
    
    float r = dist(gen);
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cum += probs[i] / sum;
        if (r < cum) return i;
    }
    
    return vocab_size - 1;
}

Tensor GPT::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad = grad_output;
    grad = final_norm_->backward(grad, learning_rate);
    
    for (int i = blocks_.size() - 1; i >= 0; --i) {
        grad = blocks_[i]->backward(grad, learning_rate);
    }
    
    return grad;
}

void GPT::update(float learning_rate) {
    for (auto& block : blocks_) {
        block->update(learning_rate);
    }
    final_norm_->update(learning_rate);
    lm_head_->update(learning_rate);
}

std::vector<Tensor> GPT::parameters() const {
    std::vector<Tensor> params = {token_embeddings_};
    for (auto& block : blocks_) {
        auto p = block->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    auto p2 = final_norm_->parameters();
    auto p3 = lm_head_->parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    return params;
}

std::vector<int> GPT::generate(const std::vector<int>& prompt, int max_tokens, float temperature) {
    std::vector<int> tokens = prompt;
    
    std::cout << "Generating tokens..." << std::endl;
    
    for (int i = 0; i < max_tokens; ++i) {
        try {
            int seq_len = std::min((int)tokens.size(), max_len_);
            int start = tokens.size() - seq_len;
            
            Tensor input({1, seq_len});
            for (int j = 0; j < seq_len; ++j) {
                input.at({0, j}) = tokens[start + j];
            }
            
            Tensor hidden = forward(input);
            Tensor logits = get_logits(hidden);
            
            Tensor last_logits({1, vocab_size_});
            for (int v = 0; v < vocab_size_; ++v) {
                last_logits.at({0, v}) = logits.at({0, seq_len - 1, v});
            }
            
            int next_token = sample_token(last_logits, temperature);
            tokens.push_back(next_token);
            
            if ((i + 1) % 5 == 0) {
                std::cout << "Generated " << (i + 1) << " tokens..." << std::endl;
            }
            
            if (next_token == 0) {
                std::cout << "EOS token reached, stopping generation." << std::endl;
                break;
            }
            
            if ((int)tokens.size() > max_len_ * 2) {
                std::cout << "Max length reached, stopping generation." << std::endl;
                break;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in generation step " << i << ": " << e.what() << std::endl;
            break;
        }
    }
    
    return tokens;
}