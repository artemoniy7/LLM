#pragma once

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>

struct AppConfig {
    std::string dataset_path = "data/pippa.jsonl";
    std::string dataset_format = "pippa_jsonl";
    std::string checkpoint_path = "model.ckpt";

    int vocab_size = 0;
    int d_model = 64;
    int num_heads = 4;
    int num_layers = 4;
    int d_ff = 128;
    int max_len = 80;

    float learning_rate = 0.001f;
    int epochs = 50;
    int max_samples = 100;
    int print_every = 5;
    int test_every = 10;

    float temperature = 0.8f;
    int max_response_tokens = 40;
    std::string system_prompt = "You are a helpful assistant for roleplay. Respond in character.";
};

inline AppConfig load_config(const std::string& filename = "config.json") {
    AppConfig config;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cout << "Config file " << filename << " not found; using built-in defaults.\n";
        return config;
    }

    nlohmann::json json;
    file >> json;

    config.dataset_path = json.value("dataset_path", config.dataset_path);
    config.dataset_format = json.value("dataset_format", config.dataset_format);
    config.checkpoint_path = json.value("checkpoint_path", config.checkpoint_path);

    if (json.contains("model")) {
        const auto& model = json["model"];
        config.vocab_size = model.value("vocab_size", config.vocab_size);
        config.d_model = model.value("d_model", config.d_model);
        config.num_heads = model.value("num_heads", config.num_heads);
        config.num_layers = model.value("num_layers", config.num_layers);
        config.d_ff = model.value("d_ff", config.d_ff);
        config.max_len = model.value("max_len", config.max_len);
    }

    if (json.contains("training")) {
        const auto& training = json["training"];
        config.learning_rate = training.value("learning_rate", config.learning_rate);
        config.epochs = training.value("epochs", config.epochs);
        config.max_samples = training.value("max_samples", config.max_samples);
        config.print_every = training.value("print_every", config.print_every);
        config.test_every = training.value("test_every", config.test_every);
    }

    if (json.contains("chat")) {
        const auto& chat = json["chat"];
        config.temperature = chat.value("temperature", config.temperature);
        config.max_response_tokens = chat.value("max_response_tokens", config.max_response_tokens);
        config.system_prompt = chat.value("system_prompt", config.system_prompt);
    }

    return config;
}
