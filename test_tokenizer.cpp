#include "include/tokenizer.hpp"
#include "include/transformer.hpp"
#include <iostream>

int main() {
    std::cout << "=== Testing Tokenizer and Model ===\n\n";
    
    SimpleTokenizer tokenizer;
    int vocab_size = tokenizer.vocab_size();
    int d_model = 32;
    int num_heads = 4;
    int num_layers = 2;
    int d_ff = 64;
    int max_len = 20;
    
    std::cout << "Vocabulary size: " << vocab_size << "\n";
    std::cout << "Creating model...\n";
    
    GPT model(vocab_size, d_model, num_heads, num_layers, d_ff, max_len);
    
    std::string prompt_text = "hello world how are you";
    std::cout << "Prompt: " << prompt_text << "\n";
    
    std::vector<int> prompt = tokenizer.encode(prompt_text);
    std::cout << "Encoded prompt: ";
    for (int t : prompt) std::cout << t << " ";
    std::cout << "\n";
    
    std::cout << "Generating...\n";
    std::vector<int> generated = model.generate(prompt, 20, 0.8f);
    
    std::string output = tokenizer.decode(generated);
    std::cout << "Generated: " << output << "\n";
    system("pause");
    return 0;
}