#include "transformer.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=== Testing Transformer Components ===\n\n";

    // Уменьшаем параметры для тестирования
    int vocab_size = 50;
    int d_model = 32;
    int num_heads = 4;
    int num_layers = 2;
    int d_ff = 64;
    int max_len = 20;

    std::cout << "Creating model with parameters:\n";
    std::cout << "  Vocabulary size: " << vocab_size << "\n";
    std::cout << "  Model dimension: " << d_model << "\n";
    std::cout << "  Number of heads: " << num_heads << "\n";
    std::cout << "  Number of layers: " << num_layers << "\n";
    std::cout << "  Feed-forward dimension: " << d_ff << "\n";
    std::cout << "  Max sequence length: " << max_len << "\n";
    std::cout << std::endl;

    GPT model(vocab_size, d_model, num_heads, num_layers, d_ff, max_len);

    std::cout << "Model created successfully!\n\n";

    // Тестовый промпт
    std::vector<int> prompt = {1, 2, 3, 4, 5};

    std::cout << "Prompt tokens: ";
    for (int token : prompt) {
        std::cout << token << " ";
    }
    std::cout << "\n";
    std::cout << "Generating text from prompt (15 tokens max)...\n" << std::endl;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> generated = model.generate(prompt, 15, 0.8f);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\nGenerated " << generated.size() << " tokens:\n";
        for (size_t i = 0; i < generated.size(); ++i) {
            std::cout << generated[i] << " ";
            if ((i + 1) % 20 == 0) std::cout << "\n";
        }
        std::cout << "\n\n";
        std::cout << "Generation time: " << duration.count() << " ms\n";
        if (duration.count() > 0) {
            std::cout << "Tokens per second: " << (generated.size() * 1000.0f / duration.count()) << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during generation: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nTest completed successfully!\n";
    return 0;
}