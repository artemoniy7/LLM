#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iterator>

class SimpleTokenizer {
public:
    SimpleTokenizer() {
        // Специальные токены
        vocab_["<PAD>"] = 0;
        vocab_["<UNK>"] = 1;
        vocab_["<BOS>"] = 2;
        vocab_["<EOS>"] = 3;
        
        // Распространенные английские слова
        std::vector<std::string> common_words = {
            "hello", "world", "how", "are", "you", "i", "am", "good", "bad", "yes", "no",
            "thanks", "please", "help", "me", "with", "this", "that", "from", "for", "my",
            "your", "what", "is", "the", "to", "of", "and", "a", "in", "it", "on", "as",
            "be", "by", "at", "or", "an", "but", "not", "so", "then", "than", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should", "may",
            "might", "must", "can", "shall", "about", "after", "all", "also", "any",
            "back", "because", "been", "before", "being", "both", "but", "came", "can",
            "come", "could", "day", "did", "does", "done", "down", "each", "even", "every",
            "find", "first", "found", "get", "give", "go", "going", "good", "got", "great",
            "had", "has", "have", "he", "her", "here", "him", "his", "home", "house",
            "how", "if", "into", "just", "know", "large", "last", "left", "like", "little",
            "long", "look", "made", "make", "man", "many", "may", "me", "men", "more",
            "most", "much", "must", "my", "name", "new", "no", "now", "number", "of",
            "off", "old", "on", "one", "only", "or", "other", "our", "out", "over",
            "own", "part", "people", "place", "point", "put", "right", "same", "say",
            "see", "she", "show", "small", "so", "some", "state", "still", "such", "take",
            "tell", "than", "that", "the", "their", "them", "then", "there", "these",
            "they", "thing", "think", "this", "those", "through", "time", "to", "too",
            "turn", "two", "up", "us", "use", "very", "want", "way", "we", "well", "went",
            "were", "what", "when", "where", "which", "while", "who", "will", "with",
            "work", "world", "would", "year", "yes", "you", "your", "young", "old",
            "friend", "love", "happy", "sad", "angry", "scared", "tired", "hungry",
            "thirsty", "cold", "hot", "warm", "cool", "beautiful", "handsome", "pretty",
            "ugly", "kind", "mean", "nice", "rude", "polite", "smart", "stupid", "funny",
            "serious", "quiet", "loud", "strong", "weak", "fast", "slow", "big", "small",
            "tall", "short", "long", "wide", "narrow", "deep", "shallow", "clear", "dark",
            "light", "bright", "dim", "soft", "hard", "smooth", "rough", "wet", "dry",
            "clean", "dirty", "easy", "difficult", "simple", "complex", "cheap", "expensive"
        };
        
        int idx = 4;
        for (const auto& word : common_words) {
            if (vocab_.find(word) == vocab_.end()) {
                vocab_[word] = idx++;
            }
        }
        
        // Создаем обратный словарь
        for (const auto& pair : vocab_) {
            inv_vocab_[pair.second] = pair.first;
        }
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        std::istringstream iss(text);
        std::string word;
        
        tokens.push_back(vocab_["<BOS>"]);
        
        while (iss >> word) {
            // Приводим к нижнему регистру
            std::transform(word.begin(), word.end(), word.begin(), 
                          [](unsigned char c) { return std::tolower(c); });
            
            // Удаляем пунктуацию
            word.erase(std::remove_if(word.begin(), word.end(), 
                                     [](unsigned char c) { return std::ispunct(c); }), 
                       word.end());
            
            if (!word.empty() && vocab_.find(word) != vocab_.end()) {
                tokens.push_back(vocab_[word]);
            } else if (!word.empty()) {
                tokens.push_back(vocab_["<UNK>"]);
            }
        }
        
        tokens.push_back(vocab_["<EOS>"]);
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string text;
        for (int token : tokens) {
            if (inv_vocab_.find(token) != inv_vocab_.end()) {
                std::string word = inv_vocab_[token];
                if (word != "<PAD>" && word != "<UNK>" && 
                    word != "<BOS>" && word != "<EOS>") {
                    text += word + " ";
                }
            }
        }
        if (!text.empty() && text.back() == ' ') {
            text.pop_back();
        }
        return text;
    }
    
    int vocab_size() const { return vocab_.size(); }
    
private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> inv_vocab_;
};