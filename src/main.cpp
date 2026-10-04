#include "model.hpp"
#include "optim.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rp;
namespace fs = std::filesystem;

static fs::path resolve_path(const std::string&input, const char*argv0) {
    fs::path requested(input);
    if (fs::exists(requested)) return fs::absolute(requested);

    fs::path exe = fs::absolute(fs::path(argv0)).parent_path();
    std::vector<fs::path> candidates = {
        requested,
        exe / requested,
        exe / ".." / requested,
        exe / ".." / ".." / requested,
        exe / ".." / ".." / ".." / requested
    };

    for (const auto& p : candidates) {
        std::error_code ec;
        fs::path normalized = fs::weakly_canonical(p, ec);
        if (!ec && fs::exists(normalized)) return normalized;
    }
    return {};
}

static std::vector<int> read_all(const fs::path& file, Tokenizer& tok) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open dataset: " + file.string());
    std::stringstream ss;
    ss << in.rdbuf();
    if (ss.fail()) throw std::runtime_error("cannot read dataset: " + file.string());
    return tok.encode(ss.str());
}

static int sample(const Tensor& logits, int row, float temp, std::mt19937& rng) {
    int V = logits->s[1];
    std::vector<float> p(V);
    float mx = -1e30f;
    for (int j = 0; j < V; ++j) mx = std::max(mx, logits->x[row * V + j]);

    float z = 0.0f;
    for (int j = 0; j < V; ++j) {
        p[j] = std::exp((logits->x[row * V + j] - mx) / std::max(0.05f, temp));
        z += p[j];
    }
    for (float& v : p) v /= z;

    std::discrete_distribution<int> d(p.begin(), p.end());
    return d(rng);
}

static void print_usage() {
    std::cout
        << "rp_llm - tiny C++ RP language model\n\n"
        << "Usage:\n"
        << "  rp_llm train [dataset] [steps] [checkpoint]\n"
        << "  rp_llm generate <checkpoint> <prompt>\n\n"
        << "Examples:\n"
        << "  rp_llm train data/rp.txt 1000 model.bin\n"
        << "  rp_llm generate model.bin \"SYSTEM: Roleplay. USER: Hello. ASSISTANT:\"\n";
}

int main(int argc, char** argv) {
    try {
        Config cfg;
        Tokenizer tok;
        Model model(cfg, 123);

        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            print_usage();
            return 0;
        }

        const std::string mode = argv[1];

        if (mode == "train") {
            const std::string data_arg = argc > 2 ? argv[2] : "data/rp.txt";
            const int steps = argc > 3 ? std::stoi(argv[3]) : 1000;
            const std::string checkpoint_arg = argc > 4 ? argv[4] : "model.bin";

            if (steps <= 0) {
                std::cerr << "Error: steps must be greater than zero.\n";
                return 2;
            }

            const fs::path data = resolve_path(data_arg, argv[0]);
            if (data.empty()) {
                std::cerr
                    << "Error: dataset not found: " << data_arg << "\n"
                    << "Current directory: " << fs::current_path().string() << "\n"
                    << "Tip: in Visual Studio set Debugging > Working Directory to the project root, "
                    << "or pass the full path to data/rp.txt.\n";
                return 3;
            }

            auto ids = read_all(data, tok);
            if (ids.size() <= static_cast<size_t>(cfg.ctx + 1)) {
                std::cerr
                    << "Error: dataset is too small (" << ids.size() << " tokens; need more than "
                    << cfg.ctx + 1 << ").\n"
                    << "Dataset: " << data.string() << "\n";
                return 4;
            }

            const fs::path checkpoint = checkpoint_arg;
            auto p = model.params();
            AdamW opt;
            opt.init(p);
            std::mt19937 rng(7);

            auto t0 = std::chrono::steady_clock::now();
            for (int s = 1; s <= steps; ++s) {
                std::uniform_int_distribution<size_t> d(0, ids.size() - cfg.ctx - 1);
                size_t at = d(rng);
                std::vector<int> x(ids.begin() + at, ids.begin() + at + cfg.ctx);
                std::vector<int> y(ids.begin() + at + 1, ids.begin() + at + cfg.ctx + 1);

                auto logits = model.forward(x);
                float loss = loss_and_seed(logits, y);
                backward(logits);
                clip(p, 1.0f);
                opt.update(p);

                if (s % 10 == 0) {
                    std::cout << "step " << s << " loss " << loss << "\n";
                    if (s % 100 == 0) model.save(checkpoint.string());
                }
            }

            model.save(checkpoint.string());
            auto dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            std::cout << "dataset: " << data.string() << "\n"
                      << "saved: " << checkpoint.string() << "\n"
                      << "time: " << dt << "s\n";
            return 0;
        }

        if (mode == "generate") {
            if (argc < 4) {
                std::cerr << "Error: generate requires <checkpoint> and <prompt>.\n";
                print_usage();
                return 2;
            }

            const std::string checkpoint = argv[2];
            const std::string prompt = argv[3];

            if (!model.load(checkpoint)) {
                std::cerr << "Error: checkpoint load failed: " << checkpoint << "\n";
                return 5;
            }

            auto ids = tok.encode(prompt);
            std::mt19937 rng(9);

            for (int n = 0; n < 120; ++n) {
                if (ids.size() > static_cast<size_t>(cfg.ctx)) ids.erase(ids.begin());
                auto logits = model.forward(ids);
                int next = sample(logits, static_cast<int>(ids.size()) - 1, 0.8f, rng);
                ids.push_back(next);
                if (next == tok.eos) break;
            }

            std::cout << tok.decode(ids) << "\n";
            return 0;
        }

        std::cerr << "Error: unknown command: " << mode << "\n";
        print_usage();
        return 2;
    } catch (const std::invalid_argument&) {
        std::cerr << "Error: invalid numeric argument.\n";
        return 2;
    } catch (const std::out_of_range&) {
        std::cerr << "Error: numeric argument is out of range.\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 10;
    }
}
