#include "config.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "=== LLM Chat ===\n\n";

    try {
        AppConfig config = load_config();
        SimpleTokenizer tokenizer;
        int vocab_size = config.vocab_size > 0 ? config.vocab_size : tokenizer.vocab_size();

        std::cout << "Checkpoint: " << config.checkpoint_path << "\n";
        GPT model(vocab_size, config.d_model, config.num_heads, config.num_layers, config.d_ff, config.max_len);
        if (model.load(config.checkpoint_path)) {
            std::cout << "Checkpoint loaded.\n";
        } else {
            std::cout << "No checkpoint found. Chat will use random weights until you run training.\n";
        }

        std::cout << "Type your messages. Type 'exit' to quit, 'clear' to clear history.\n\n";

        std::vector<int> history = tokenizer.encode(config.system_prompt);
        std::string input;

        while (true) {
            std::cout << "You: ";
            if (!std::getline(std::cin, input)) {
                break;
            }

            if (input == "exit") {
                break;
            }
            if (input == "clear") {
                history = tokenizer.encode(config.system_prompt);
                std::cout << "History cleared.\n\n";
                continue;
            }

            std::vector<int> user_tokens = tokenizer.encode("User: " + input);
            history.insert(history.end(), user_tokens.begin(), user_tokens.end());

            int seq_len = std::min(static_cast<int>(history.size()), config.max_len);
            int start = static_cast<int>(history.size()) - seq_len;
            std::vector<int> prompt(history.begin() + start, history.end());

            std::vector<int> response = model.generate(prompt, config.max_response_tokens, config.temperature);
            std::string response_text = tokenizer.decode(response);
            std::cout << "Assistant: " << response_text << "\n\n";

            history.insert(history.end(), response.begin(), response.end());
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Goodbye!\n";
    return 0;
}
