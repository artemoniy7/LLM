#include "config.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <cctype>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct Dialogue {
    std::vector<std::pair<std::string, std::string>> turns;
};

static std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

static void add_turn(Dialogue& dialogue, const std::string& role, const std::string& text) {
    std::string cleaned = trim(text);
    if (!cleaned.empty()) {
        dialogue.turns.push_back({role, cleaned});
    }
}

static std::vector<std::pair<std::string, std::string>> parse_pippa_definitions(const std::string& definitions,
                                                                                 int max_lines = 8) {
    std::vector<std::pair<std::string, std::string>> turns;
    std::istringstream stream(definitions);
    std::string line;

    while (std::getline(stream, line) && static_cast<int>(turns.size()) < max_lines) {
        size_t user_pos = line.find("{{user}}:");
        size_t char_pos = line.find("{{char}}:");

        if (user_pos != std::string::npos) {
            turns.push_back({"user", trim(line.substr(user_pos + 9))});
        } else if (char_pos != std::string::npos) {
            turns.push_back({"assistant", trim(line.substr(char_pos + 9))});
        }
    }

    return turns;
}

static std::vector<Dialogue> load_pippa_jsonl(const std::string& filename, int max_samples) {
    std::vector<Dialogue> dialogues;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: could not open dataset " << filename << "\n";
        return dialogues;
    }

    while (std::getline(file, line) && static_cast<int>(dialogues.size()) < max_samples) {
        if (trim(line).empty()) {
            continue;
        }

        try {
            nlohmann::json row = nlohmann::json::parse(line);
            Dialogue dialogue;

            add_turn(dialogue, "assistant", row.value("bot_greeting", ""));
            for (const auto& turn : parse_pippa_definitions(row.value("bot_definitions", ""))) {
                add_turn(dialogue, turn.first, turn.second);
            }

            if (!dialogue.turns.empty()) {
                dialogues.push_back(dialogue);
            }
        } catch (const std::exception&) {
            continue;
        }
    }

    return dialogues;
}

static std::vector<Dialogue> load_pipe_dialogues(const std::string& filename, int max_samples) {
    std::vector<Dialogue> dialogues;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: could not open dataset " << filename << "\n";
        return dialogues;
    }

    while (std::getline(file, line) && static_cast<int>(dialogues.size()) < max_samples) {
        size_t separator = line.find('|');
        if (separator == std::string::npos) {
            continue;
        }

        Dialogue dialogue;
        add_turn(dialogue, "user", line.substr(0, separator));
        add_turn(dialogue, "assistant", line.substr(separator + 1));
        if (!dialogue.turns.empty()) {
            dialogues.push_back(dialogue);
        }
    }

    return dialogues;
}

static std::vector<Dialogue> load_dataset(const AppConfig& config) {
    if (config.dataset_format == "pipe") {
        return load_pipe_dialogues(config.dataset_path, config.max_samples);
    }
    return load_pippa_jsonl(config.dataset_path, config.max_samples);
}

static std::vector<Tensor> prepare_data(const std::vector<Dialogue>& dialogues,
                                        SimpleTokenizer& tokenizer,
                                        int max_len) {
    std::vector<Tensor> samples;

    for (const auto& dialogue : dialogues) {
        std::string text;
        for (const auto& turn : dialogue.turns) {
            text += (turn.first == "user" ? "User: " : "Assistant: ");
            text += turn.second;
            text += ' ';
        }

        std::vector<int> tokens = tokenizer.encode(text);
        if (static_cast<int>(tokens.size()) > max_len) {
            tokens.resize(max_len);
        }
        if (tokens.size() < 3) {
            continue;
        }

        Tensor input({1, static_cast<int>(tokens.size())});
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
            input.at({0, i}) = tokens[i];
        }
        samples.push_back(input);
    }

    return samples;
}

static void shuffle_data(std::vector<Tensor>& data) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(data.begin(), data.end(), generator);
}

static void print_sample_generation(GPT& model, SimpleTokenizer& tokenizer, const AppConfig& config) {
    std::vector<int> prompt = tokenizer.encode("User: hello");
    std::vector<int> response = model.generate(prompt, 10, config.temperature);
    std::cout << "  Prompt: User: hello\n";
    std::cout << "  Response: " << tokenizer.decode(response) << "\n\n";
}

int main() {
    std::cout << "=== LLM Training ===\n\n";

    try {
        AppConfig config = load_config();
        SimpleTokenizer tokenizer;
        int vocab_size = config.vocab_size > 0 ? config.vocab_size : tokenizer.vocab_size();

        std::cout << "Dataset: " << config.dataset_path << " (" << config.dataset_format << ")\n";
        std::cout << "Checkpoint: " << config.checkpoint_path << "\n";
        std::cout << "Model: vocab=" << vocab_size
                  << ", d_model=" << config.d_model
                  << ", heads=" << config.num_heads
                  << ", layers=" << config.num_layers
                  << ", d_ff=" << config.d_ff
                  << ", max_len=" << config.max_len << "\n\n";

        GPT model(vocab_size, config.d_model, config.num_heads, config.num_layers, config.d_ff, config.max_len);
        if (model.load(config.checkpoint_path)) {
            std::cout << "Checkpoint loaded; continuing training.\n";
        } else {
            std::cout << "No checkpoint found; training from scratch.\n";
        }

        std::vector<Dialogue> dialogues = load_dataset(config);
        std::cout << "Loaded " << dialogues.size() << " dialogues\n";
        if (dialogues.empty()) {
            std::cerr << "No training data loaded. Check config.json dataset_path and dataset_format.\n";
            return 1;
        }

        std::vector<Tensor> data = prepare_data(dialogues, tokenizer, config.max_len);
        std::cout << "Prepared " << data.size() << " samples\n";
        if (data.empty()) {
            std::cerr << "No usable samples prepared.\n";
            return 1;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        float best_loss = std::numeric_limits<float>::max();

        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            float total_loss = 0.0f;
            int batches = 0;
            shuffle_data(data);

            for (const auto& input : data) {
                try {
                    int seq_len = input.shape()[1];
                    Tensor hidden = model.forward(input);
                    Tensor logits = model.get_logits(hidden);

                    float sample_loss = 0.0f;
                    int valid_positions = 0;
                    int last_valid_pos = -1;

                    for (int pos = 0; pos < seq_len - 1; ++pos) {
                        int target_token = static_cast<int>(input.at({0, pos + 1}));
                        if (target_token == 0 || target_token == 1 || target_token == 3) {
                            continue;
                        }

                        Tensor pred({1, vocab_size});
                        for (int v = 0; v < vocab_size; ++v) {
                            pred.at({0, v}) = logits.at({0, pos, v});
                        }

                        Tensor target({1, 1});
                        target.at({0, 0}) = target_token;
                        CrossEntropyLoss loss_fn;
                        sample_loss += loss_fn.forward(pred, target);
                        valid_positions++;
                        last_valid_pos = pos;
                    }

                    if (valid_positions == 0) {
                        continue;
                    }

                    total_loss += sample_loss / valid_positions;
                    batches++;

                    int target_token = static_cast<int>(input.at({0, last_valid_pos + 1}));
                    Tensor pred({1, vocab_size});
                    for (int v = 0; v < vocab_size; ++v) {
                        pred.at({0, v}) = logits.at({0, last_valid_pos, v});
                    }

                    Tensor target({1, 1});
                    target.at({0, 0}) = target_token;
                    CrossEntropyLoss loss_fn;
                    loss_fn.forward(pred, target);
                    Tensor grad = loss_fn.backward();

                    Tensor grad_3d({1, seq_len, vocab_size}, 0.0f);
                    for (int v = 0; v < vocab_size; ++v) {
                        grad_3d.at({0, last_valid_pos, v}) = grad.at({0, v});
                    }

                    model.backward(grad_3d, config.learning_rate);
                    model.update(config.learning_rate);
                } catch (const std::exception&) {
                    continue;
                }
            }

            if ((epoch + 1) % config.print_every == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
                float avg_loss = batches > 0 ? total_loss / batches : 0.0f;

                if (avg_loss < best_loss) {
                    best_loss = avg_loss;
                    model.save(config.checkpoint_path);
                    std::cout << "Saved best checkpoint.\n";
                }

                std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs
                          << " | Loss: " << avg_loss
                          << " | Best: " << best_loss
                          << " | Batches: " << batches
                          << " | Time: " << elapsed.count() << "s\n";

                if (config.test_every > 0 && (epoch + 1) % config.test_every == 0) {
                    print_sample_generation(model, tokenizer, config);
                }
            }
        }

        model.save(config.checkpoint_path);
        std::cout << "Training complete. Final checkpoint saved to " << config.checkpoint_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
