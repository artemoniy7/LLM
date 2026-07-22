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
#include <sstream>
#include "json.hpp"

using json = nlohmann::json;

// Структура для диалога
struct Dialogue {
    std::string bot_name;
    std::string bot_greeting;
    std::string bot_definitions;
    std::vector<std::pair<std::string, std::string>> turns; // user -> bot
};

// Парсинг определений бота
std::vector<std::pair<std::string, std::string>> parse_definitions(const std::string& definitions, int max_lines = 6) {
    std::vector<std::pair<std::string, std::string>> turns;
    std::istringstream iss(definitions);
    std::string line;
    int count = 0;

    while (std::getline(iss, line) && count < max_lines) {
        if (line.empty()) continue;

        size_t user_pos = line.find("{{user}}:");
        size_t char_pos = line.find("{{char}}:");

        if (user_pos != std::string::npos) {
            std::string user_text = line.substr(user_pos + 8);
            while (!user_text.empty() && user_text.front() == ' ') user_text.erase(0, 1);
            turns.push_back({"user", user_text});
            count++;
        } else if (char_pos != std::string::npos) {
            std::string char_text = line.substr(char_pos + 8);
            while (!char_text.empty() && char_text.front() == ' ') char_text.erase(0, 1);
            turns.push_back({"bot", char_text});
            count++;
        }
    }

    return turns;
}

// Загрузка датасета из JSONL (только первые N диалогов)
std::vector<Dialogue> load_pippa_dataset(const std::string& filename, int max_samples = 10) {
    std::vector<Dialogue> dialogues;
    std::ifstream file(filename);
    std::string line;
    int count = 0;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return dialogues;
    }

    while (std::getline(file, line) && count < max_samples) {
        if (line.empty()) continue;

        try {
            json data = json::parse(line);

            Dialogue dialog;
            dialog.bot_name = data.value("bot_name", "Unknown");
            dialog.bot_greeting = data.value("bot_greeting", "");
            dialog.bot_definitions = data.value("bot_definitions", "");

            auto turns = parse_definitions(dialog.bot_definitions, 6);

            if (!dialog.bot_greeting.empty()) {
                turns.insert(turns.begin(), {"bot", dialog.bot_greeting});
            }

            if (!turns.empty()) {
                if (turns.size() > 8) turns.resize(8);
                dialog.turns = turns;
                dialogues.push_back(dialog);
                count++;
            }

        } catch (const std::exception& e) {
            continue;
        }
    }

    return dialogues;
}

// Подготовка данных для обучения
std::vector<Tensor> prepare_dialogue_data(const std::vector<Dialogue>& dialogues,
                                          SimpleTokenizer& tokenizer,
                                          int max_len) {
    std::vector<Tensor> data;

    for (const auto& dialog : dialogues) {
        try {
            std::string full_text = "";

            for (const auto& turn : dialog.turns) {
                if (turn.first == "user") {
                    full_text += "User: " + turn.second + " ";
                } else {
                    full_text += "Assistant: " + turn.second + " ";
                }
            }

            std::vector<int> tokens = tokenizer.encode(full_text);

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

int main() {
    std::cout << "=== Training LLM on PIPPA (FAST) ===\n\n";

    try {
        const std::string dataset_path = "data/pippa.jsonl";
        const std::string checkpoint_path = "pippa_model.ckpt";

        // Уменьшаем параметры для скорости
        int vocab_size = 1000;
        int d_model = 64;
        int num_heads = 4;
        int num_layers = 5;
        int d_ff = 128;
        int max_len = 120;

        std::cout << "Creating model (small for speed)...\n";
        std::cout << "  d_model: " << d_model << "\n";
        std::cout << "  num_layers: " << num_layers << "\n";
        std::cout << "  d_ff: " << d_ff << "\n";
        std::cout << "  max_len: " << max_len << "\n\n";

        GPT model(vocab_size, d_model, num_heads, num_layers, d_ff, max_len);
        if (model.load(checkpoint_path)) {
            std::cout << "Checkpoint loaded from " << checkpoint_path << " — continuing training.\n";
        } else {
            std::cout << "No checkpoint found; training from scratch.\n";
        }
        std::cout << "Model ready!\n";

        SimpleTokenizer tokenizer;
        std::cout << "Tokenizer vocab size: " << tokenizer.vocab_size() << "\n";

        // Загружаем пример PIPPA из data/pippa.jsonl
        std::cout << "\nLoading PIPPA dataset from " << dataset_path << "...\n";

        auto dialogues = load_pippa_dataset(dataset_path, 30);
        std::cout << "Loaded " << dialogues.size() << " dialogues\n";

        if (dialogues.empty()) {
            std::cerr << "Error: No dialogues loaded!\n";
            std::cerr << "Make sure the file exists at: " << dataset_path << "\n";
            return 1;
        }

        // Показываем пример
        if (!dialogues.empty()) {
            std::cout << "\nExample dialogue:\n";
            std::cout << "Bot: " << dialogues[0].bot_name << "\n";
            int show = 0;
            for (const auto& turn : dialogues[0].turns) {
                if (turn.first == "user") {
                    std::cout << "  User: " << turn.second << "\n";
                } else {
                    std::cout << "  Assistant: " << turn.second << "\n";
                }
                if (++show >= 4) break;
            }
            std::cout << "\n";
        }

        // Подготовка данных
        std::cout << "Preparing data...\n";
        auto data = prepare_dialogue_data(dialogues, tokenizer, max_len);
        std::cout << "Prepared " << data.size() << " samples\n";

        if (data.empty()) {
            std::cerr << "Error: No data prepared!\n";
            return 1;
        }

        // Обучение (быстрое)
        std::cout << "\nStarting training (fast mode, 30 epochs)...\n\n";

        float learning_rate = 0.005f;
        int epochs = 80;
        int print_every = 5;
        int test_every = 10;

        auto start_time = std::chrono::high_resolution_clock::now();
        float best_loss = 100.0f;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int batches = 0;

            // Перемешиваем
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(data.begin(), data.end(), g);

            for (const auto& input : data) {
                try {
                    int seq_len = input.shape()[1];

                    // Forward pass
                    Tensor hidden = model.forward(input);
                    Tensor logits = model.get_logits(hidden);

                    float sample_loss = 0.0f;
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
                        sample_loss += loss_fn.forward(pred, targ);
                        valid++;
                    }

                    if (valid == 0) continue;

                    float avg_loss = sample_loss / valid;
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
                    continue;
                }
            }

            if ((epoch + 1) % print_every == 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

                float avg_loss = (batches > 0) ? total_loss / batches : 0.0f;
                if (avg_loss < best_loss) {
                    best_loss = avg_loss;
                    model.save(checkpoint_path);
                    std::cout << "Saved best checkpoint to " << checkpoint_path << "\n";
                }

                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                          << " | Loss: " << avg_loss
                          << " | Best: " << best_loss
                          << " | Time: " << duration.count() << "s\n";

                if ((epoch + 1) % test_every == 0) {
                    std::cout << "  Testing: ";
                    try {
                        std::vector<int> prompt = tokenizer.encode("User: hello");
                        std::vector<int> response = model.generate(prompt, 8, 0.7f);
                        std::string text = tokenizer.decode(response);
                        std::cout << text << "\n";
                    } catch (const std::exception& e) {
                        std::cout << "Error\n";
                    }
                }
            }
        }

        std::cout << "\nTraining complete!\n";
        std::cout << "Best loss: " << best_loss << "\n";
        model.save(checkpoint_path);
        std::cout << "Final checkpoint saved to " << checkpoint_path << "\n";

        // Финальный тест
        std::cout << "\n=== Final Tests ===\n";
        std::vector<std::string> tests = {
            "hello",
            "how are you",
            "what is your name"
        };

        for (const auto& test : tests) {
            try {
                std::vector<int> prompt = tokenizer.encode("User: " + test);
                std::vector<int> response = model.generate(prompt, 12, 0.7f);
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

    return 0;
}