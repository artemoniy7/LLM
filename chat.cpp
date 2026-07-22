#include "transformer.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <string>

int main() {
    std::cout << "=== RP Chat Interface ===\n\n";
    
    // Параметры модели
    int vocab_size = 200;
    int d_model = 64;
    int num_heads = 4;
    int num_layers = 3;
    int d_ff = 128;
    int max_len = 50;
    
    std::cout << "Loading model...\n";
    GPT model(vocab_size, d_model, num_heads, num_layers, d_ff, max_len);
    SimpleTokenizer tokenizer;
    
    std::cout << "Model loaded!\n";
    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << "\n\n";
    
    std::cout << "=== RP Chat ===\n";
    std::cout << "Type your messages. Type 'exit' to quit.\n";
    std::cout << "Type 'clear' to clear history.\n\n";
    
    std::vector<int> history;
    float temperature = 0.8f;
    int max_response_tokens = 30;
    
    // Системный промпт для RP
    std::string system_prompt = "You are a helpful assistant for roleplay. Respond in character.";
    std::vector<int> system_tokens = tokenizer.encode(system_prompt);
    history.insert(history.end(), system_tokens.begin(), system_tokens.end());
    
    std::string input;
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        if (input == "exit") {
            break;
        }
        if (input == "clear") {
            history.clear();
            history.insert(history.end(), system_tokens.begin(), system_tokens.end());
            std::cout << "History cleared.\n\n";
            continue;
        }
        
        // Токенизируем ввод
        std::vector<int> input_tokens = tokenizer.encode(input);
        history.insert(history.end(), input_tokens.begin(), input_tokens.end());
        
        // Генерируем ответ
        std::cout << "Assistant: ";
        std::cout.flush();
        
        try {
            // Берем последние max_len токенов для генерации
            int seq_len = std::min((int)history.size(), max_len);
            int start = history.size() - seq_len;
            std::vector<int> prompt(history.begin() + start, history.end());
            
            std::vector<int> response = model.generate(prompt, max_response_tokens, temperature);
            
            // Декодируем и выводим ответ
            std::string response_text = tokenizer.decode(response);
            
            // Убираем системный промпт из вывода
            size_t pos = response_text.find(system_prompt);
            if (pos != std::string::npos) {
                response_text = response_text.substr(pos + system_prompt.length());
            }
            
            std::cout << response_text << "\n\n";
            
            // Добавляем ответ в историю
            history.insert(history.end(), response.begin(), response.end());
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }
    
    std::cout << "\nGoodbye!\n";
    return 0;
}