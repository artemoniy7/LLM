#include "transformer.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <exception>

// Загрузка датасета
std::vector<std::pair<std::string, std::string>> load_dataset(const std::string& filename) {
    std::vector<std::pair<std::string, std::string>> dataset;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return dataset;
    }
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t pos = line.find('|');
        if (pos != std::string::npos) {
            std::string user = line.substr(0, pos);
            std::string assistant = line.substr(pos + 1);
            while (!user.empty() && user.back() == ' ') user.pop_back();
            while (!assistant.empty() && assistant.front() == ' ') assistant.erase(0, 1);
            dataset.push_back({user, assistant});
        }
    }
    
    return dataset;
}

// Функция для вывода размерности тензора
void print_shape(const Tensor& t, const std::string& name) {
    std::cout << name << " shape: ";
    for (int d : t.shape()) std::cout << d << " ";
    std::cout << "\n";
}

int main() {
    std::cout << "=== Training LLM for RP (Simplified) ===\n\n";
    
    try {
        // Параметры модели
        int vocab_size = 300;
        int d_model = 16;
        int num_heads = 2;
        int num_layers = 1;
        int d_ff = 32;
        int max_len = 20;
        
        std::cout << "Creating model with params:\n";
        std::cout << "  d_model: " << d_model << "\n";
        std::cout << "  num_layers: " << num_layers << "\n";
        std::cout << "  d_ff: " << d_ff << "\n";
        std::cout << "  max_len: " << max_len << "\n\n";
        
        GPT model(vocab_size, d_model, num_heads, num_layers, d_ff, max_len);
        std::cout << "Model created!\n";
        
        SimpleTokenizer tokenizer;
        std::cout << "Tokenizer vocab size: " << tokenizer.vocab_size() << "\n";
        
        std::cout << "Loading dataset...\n";
        auto dataset = load_dataset("data/dialogues.txt");
        std::cout << "Loaded " << dataset.size() << " dialogues\n";
        
        if (dataset.empty()) {
            std::cerr << "Error: No data loaded!\n";
            return 1;
        }
        
        // Подготовка данных
        std::string full_text = "";
        for (const auto& pair : dataset) {
            full_text += "User: " + pair.first + " Assistant: " + pair.second + " ";
        }
        
        std::vector<int> all_tokens = tokenizer.encode(full_text);
        std::cout << "Total tokens: " << all_tokens.size() << "\n";
        
        // Разбиваем на последовательности
        int seq_len = 8;  // еще короче
        std::vector<Tensor> sequences;
        
        for (size_t i = 0; i + seq_len < all_tokens.size(); i += seq_len) {
            try {
                Tensor seq({1, seq_len});
                for (int j = 0; j < seq_len; ++j) {
                    seq.at({0, j}) = all_tokens[i + j];
                }
                sequences.push_back(seq);
            } catch (const std::exception& e) {
                std::cerr << "Error creating sequence: " << e.what() << "\n";
                continue;
            }
        }
        
        std::cout << "Created " << sequences.size() << " sequences\n\n";
        
        if (sequences.empty()) {
            std::cerr << "Error: No sequences created!\n";
            return 1;
        }
        
        // === ТЕСТ Forward Pass ===
        std::cout << "=== Testing forward pass ===\n";
        try {
            Tensor test_input({1, 5});
            for (int i = 0; i < 5; ++i) test_input.at({0, i}) = i + 2;
            print_shape(test_input, "Test input");
            
            Tensor hidden = model.forward(test_input);
            print_shape(hidden, "Hidden");
            
            Tensor logits = model.get_logits(hidden);
            print_shape(logits, "Logits");
            
            std::cout << "Forward pass successful!\n\n";
        } catch (const std::exception& e) {
            std::cerr << "Forward pass error: " << e.what() << "\n";
            return 1;
        }
        
        // === ОБУЧЕНИЕ ===
        float learning_rate = 0.01f;
        int epochs = 30;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int batches = 0;
            
            // Перемешиваем
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(sequences.begin(), sequences.end(), g);
            
            for (const auto& input : sequences) {
                try {
                    // Forward pass
                    Tensor hidden = model.forward(input);
                    Tensor logits = model.get_logits(hidden);
                    
                    // Вычисляем loss для каждой позиции
                    float seq_loss = 0.0f;
                    int valid = 0;
                    
                    for (int pos = 0; pos < seq_len - 1; ++pos) {
                        int target = (int)input.at({0, pos + 1});
                        
                        if (target == 0 || target == 1 || target == 3) continue;
                        
                        Tensor pred({1, vocab_size});
                        for (int v = 0; v < vocab_size; ++v) {
                            pred.at({0, v}) = logits.at({0, pos, v});
                        }
                        
                        Tensor targ({1, 1});
                        targ.at({0, 0}) = target;
                        
                        CrossEntropyLoss loss_fn;
                        seq_loss += loss_fn.forward(pred, targ);
                        valid++;
                    }
                    
                    if (valid == 0) continue;
                    
                    float avg_loss = seq_loss / valid;
                    total_loss += avg_loss;
                    batches++;
                    
                    // Backward pass
                    int last_pos = seq_len - 2;
                    int target = (int)input.at({0, last_pos + 1});
                    
                    if (target != 0 && target != 1 && target != 3) {
                        Tensor pred({1, vocab_size});
                        for (int v = 0; v < vocab_size; ++v) {
                            pred.at({0, v}) = logits.at({0, last_pos, v});
                        }
                        
                        Tensor targ({1, 1});
                        targ.at({0, 0}) = target;
                        
                        CrossEntropyLoss loss_fn;
                        loss_fn.forward(pred, targ);
                        Tensor grad = loss_fn.backward();
                        
                        // Создаем градиент для всей последовательности
                        Tensor grad_3d({1, seq_len, vocab_size});
                        for (int pos = 0; pos < seq_len; ++pos) {
                            for (int v = 0; v < vocab_size; ++v) {
                                grad_3d.at({0, pos, v}) = grad.at({0, v});
                            }
                        }
                        
                        model.backward(grad_3d, learning_rate);
                        model.update(learning_rate);
                    }
                    
                } catch (const std::exception& e) {
                    // Пропускаем ошибки
                    continue;
                }
            }
            
            if ((epoch + 1) % 5 == 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
                
                float avg_loss = (batches > 0) ? total_loss / batches : 0.0f;
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                          << " | Loss: " << avg_loss
                          << " | Time: " << duration.count() << "s\n";
                
                // Тест генерации
                if ((epoch + 1) % 10 == 0) {
                    try {
                        std::vector<int> prompt = tokenizer.encode("User: hello");
                        std::vector<int> response = model.generate(prompt, 6, 0.8f);
                        std::string text = tokenizer.decode(response);
                        std::cout << "  Test: " << text << "\n";
                    } catch (const std::exception& e) {
                        std::cout << "  Error: " << e.what() << "\n";
                    }
                }
            }
        }
        
        std::cout << "\nTraining complete!\n";
        
        // Финальный тест
        std::cout << "\n=== Final Tests ===\n";
        std::vector<std::string> tests = {"hello", "how are you"};
        
        for (const auto& test : tests) {
            try {
                std::vector<int> prompt = tokenizer.encode("User: " + test);
                std::vector<int> response = model.generate(prompt, 10, 0.7f);
                std::string text = tokenizer.decode(response);
                std::cout << "User: " << test << "\n";
                std::cout << "Assistant: " << text << "\n\n";
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
    }
    
    system("pause");
    return 0;
}