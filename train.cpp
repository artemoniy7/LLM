#include "transformer.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <exception>
#include <algorithm>
#include <cmath>

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

// Подготовка данных для обучения
std::vector<Tensor> prepare_data(const std::vector<std::pair<std::string, std::string>>& dataset, 
                                  SimpleTokenizer& tokenizer, int max_len) {
    std::vector<Tensor> data;
    
    for (const auto& pair : dataset) {
        try {
            std::string text = "User: " + pair.first + " Assistant: " + pair.second;
            std::vector<int> tokens = tokenizer.encode(text);
            
            if ((int)tokens.size() > max_len) {
                tokens.resize(max_len);
            }
            
            if (tokens.size() < 3) continue;
            
            Tensor input({1, (int)tokens.size()});
            for (int i = 0; i < (int)tokens.size(); ++i) {
                input.at({0, i}) = tokens[i];
            }
            data.push_back(input);
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    return data;
}

// Перемешивание данных
void shuffle_data(std::vector<Tensor>& data) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
}

int main() {
    std::cout << "=== Training LLM for RP ===\n\n";
    
    try {
        // Параметры модели
        int vocab_size = 300;
        int d_model = 64;
        int num_heads = 4;
        int num_layers = 4;
        int d_ff = 128;
        int max_len = 50;
        
        std::cout << "Creating model with params:\n";
        std::cout << "  d_model: " << d_model << "\n";
        std::cout << "  num_layers: " << num_layers << "\n";
        std::cout << "  d_ff: " << d_ff << "\n";
        std::cout << "  max_len: " << max_len << "\n\n";
        
        GPT model(vocab_size, d_model, num_heads, num_layers, d_ff, max_len);
        std::cout << "Model created!\n";
        
        SimpleTokenizer tokenizer;
        std::cout << "Tokenizer created! Vocab size: " << tokenizer.vocab_size() << "\n";
        
        std::cout << "Loading dataset...\n";
        auto dataset = load_dataset("data/dialogues.txt");
        std::cout << "Loaded " << dataset.size() << " dialogues\n";
        
        if (dataset.empty()) {
            std::cerr << "Error: No data loaded! Create data/dialogues.txt file.\n";
            return 1;
        }
        
        std::cout << "Sample dialogues:\n";
        for (size_t i = 0; i < std::min(dataset.size(), size_t(3)); ++i) {
            std::cout << "  User: " << dataset[i].first << " -> Assistant: " << dataset[i].second << "\n";
        }
        std::cout << "\n";
        
        std::cout << "Preparing data...\n";
        auto data = prepare_data(dataset, tokenizer, max_len);
        std::cout << "Prepared " << data.size() << " samples\n";
        
        if (data.empty()) {
            std::cerr << "Error: No data prepared!\n";
            return 1;
        }
        
        std::cout << "Starting training...\n\n";
        
        float learning_rate = 0.001f;  // Уменьшили LR
        int epochs = 500;
        int print_every = 10;
        int test_every = 50;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Для отслеживания лучшего loss
        float best_loss = 100.0f;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int batches = 0;
            
            shuffle_data(data);
            
            for (size_t i = 0; i < data.size(); ++i) {
                try {
                    Tensor input = data[i];
                    int seq_len = input.shape()[1];
                    
                    // Forward pass
                    Tensor hidden = model.forward(input);
                    Tensor logits = model.get_logits(hidden);
                    
                    // Вычисляем loss для всех позиций
                    float sample_loss = 0.0f;
                    int valid_positions = 0;
                    
                    // Собираем все градиенты
                    std::vector<Tensor> gradients;
                    
                    for (int pos = 0; pos < seq_len - 1; ++pos) {
                        int target_token = (int)input.at({0, pos + 1});
                        
                        // Пропускаем специальные токены
                        if (target_token == 0 || target_token == 1 || target_token == 3) {
                            continue;
                        }
                        
                        // Создаем предсказание для текущей позиции
                        Tensor pred({1, vocab_size});
                        for (int v = 0; v < vocab_size; ++v) {
                            pred.at({0, v}) = logits.at({0, pos, v});
                        }
                        
                        Tensor targ({1, 1});
                        targ.at({0, 0}) = target_token;
                        
                        CrossEntropyLoss loss_fn;
                        float loss = loss_fn.forward(pred, targ);
                        sample_loss += loss;
                        valid_positions++;
                        
                        // Сохраняем градиент для этой позиции
                        Tensor grad = loss_fn.backward();
                        gradients.push_back(grad);
                    }
                    
                    if (valid_positions == 0) continue;
                    
                    // Усредняем loss
                    float avg_loss = sample_loss / valid_positions;
                    total_loss += avg_loss;
                    batches++;
                    
                    // Усредняем градиенты по всем позициям
                    if (!gradients.empty()) {
                        Tensor avg_grad({1, vocab_size}, 0.0f);
                        for (const auto& g : gradients) {
                            for (int v = 0; v < vocab_size; ++v) {
                                avg_grad.at({0, v}) += g.at({0, v});
                            }
                        }
                        for (int v = 0; v < vocab_size; ++v) {
                            avg_grad.at({0, v}) /= gradients.size();
                        }
                        
                        // Применяем градиент ко всем позициям
                        Tensor grad_3d({1, seq_len, vocab_size});
                        for (int pos = 0; pos < seq_len; ++pos) {
                            for (int v = 0; v < vocab_size; ++v) {
                                grad_3d.at({0, pos, v}) = avg_grad.at({0, v});
                            }
                        }
                        
                        model.backward(grad_3d, learning_rate);
                        model.update(learning_rate);
                    }
                    
                } catch (const std::exception& e) {
                    continue;
                }
            }
            
            // Выводим прогресс
            if ((epoch + 1) % print_every == 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
                
                float avg_loss = (batches > 0) ? total_loss / batches : 0.0f;
                
                // Обновляем лучший loss
                if (avg_loss < best_loss) {
                    best_loss = avg_loss;
                }
                
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                          << " | Loss: " << avg_loss
                          << " | Best: " << best_loss
                          << " | Batches: " << batches
                          << " | Time: " << duration.count() << "s\n";
                
                if ((epoch + 1) % test_every == 0) {
                    std::cout << "\nTesting generation:\n";
                    try {
                        std::vector<int> prompt = tokenizer.encode("User: hello");
                        std::vector<int> response = model.generate(prompt, 10, 0.7f);
                        std::string text = tokenizer.decode(response);
                        std::cout << "  Prompt: User: hello\n";
                        std::cout << "  Response: " << text << "\n\n";
                    } catch (const std::exception& e) {
                        std::cout << "  Error generating: " << e.what() << "\n\n";
                    }
                }
            }
        }
        
        std::cout << "\nTraining complete!\n";
        std::cout << "Best loss achieved: " << best_loss << "\n";
        
        // Тестируем финальную модель
        std::cout << "\nTesting trained model:\n";
        std::vector<std::string> test_prompts = {
            "hello",
            "how are you",
            "i need help",
            "tell me a joke",
            "goodbye"
        };
        
        for (const auto& test_prompt : test_prompts) {
            try {
                std::string full_prompt = "User: " + test_prompt;
                std::vector<int> prompt = tokenizer.encode(full_prompt);
                std::vector<int> response = model.generate(prompt, 15, 0.7f);
                std::string text = tokenizer.decode(response);
                std::cout << "  User: " << test_prompt << "\n";
                std::cout << "  Assistant: " << text << "\n\n";
            } catch (const std::exception& e) {
                std::cout << "  Error: " << e.what() << "\n\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        system("pause");
        return 1;
    }
    
    std::cout << "\nDone!\n";
    system("pause");
    return 0;
}