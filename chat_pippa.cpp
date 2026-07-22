#include "transformer.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "=== RP Chat with PIPPA Model ===\n\n";

    const std::string checkpoint_path = "pippa_model.ckpt";

    // Параметры модели (должны совпадать с train_pippa_fast.cpp)
    int vocab_size = 1000;
    int d_model = 64;
    int num_heads = 4;
    int num_layers = 5;
    int d_ff = 128;
    int max_len = 120;

    std::cout << "Loading model...\n";
    GPT model(vocab_size, d_model, num_heads, num_layers, d_ff, max_len);
    SimpleTokenizer tokenizer;
    if (model.load(checkpoint_path)) {
        std::cout << "Checkpoint loaded from " << checkpoint_path << "!\n";
    } else {
        std::cout << "No checkpoint found. Run train_pippa_fast first.\n";
    }

    std::cout << "Model loaded!\n\n";
    std::cout << "=== RP Chat ===\n";
    std::cout << "Type your messages. Type 'exit' to quit.\n";
    std::cout << "Type 'clear' to clear history.\n\n";

    // Системный промпт
    std::string system_prompt = "You are a friendly assistant for roleplay. Respond in character.";
    std::vector<int> history = tokenizer.encode(system_prompt);

    std::string input;
    float temperature = 0.8f;

    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, input);

        if (input == "exit") break;
        if (input == "clear") {
            history = tokenizer.encode(system_prompt);
            std::cout << "History cleared.\n\n";
            continue;
        }

        // Добавляем сообщение пользователя
        std::vector<int> user_tokens = tokenizer.encode("User: " + input);
        history.insert(history.end(), user_tokens.begin(), user_tokens.end());

        // Генерируем ответ
        std::cout << "Assistant: ";
        std::cout.flush();

        try {
            int seq_len = std::min((int)history.size(), max_len);
            int start = history.size() - seq_len;
            std::vector<int> prompt(history.begin() + start, history.end());

            std::vector<int> response = model.generate(prompt, 30, temperature);
            std::string text = tokenizer.decode(response);

            // Убираем системный промпт
            size_t pos = text.find(system_prompt);
            if (pos != std::string::npos) {
                text = text.substr(pos + system_prompt.length());
            }

            std::cout << text << "\n\n";

            // Добавляем ответ в историю
            history.insert(history.end(), response.begin(), response.end());

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }

    std::cout << "\nGoodbye!\n";
    return 0;
}