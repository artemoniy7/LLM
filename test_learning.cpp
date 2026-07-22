#include "layer.hpp"
#include <iostream>
#include <memory>
#include <cmath>

int main() {
    std::cout << "=== Training a Simple Neural Network ===\n\n";
    
    // Улучшенная модель: 2 входа -> 8 нейронов -> Tanh -> 4 нейрона -> ReLU -> 2 выхода
    Sequential model;
    model.add_layer(std::make_shared<Linear>(2, 8));     // больше нейронов
    model.add_layer(std::make_shared<Tanh>());           // Tanh вместо ReLU для первого слоя
    model.add_layer(std::make_shared<Linear>(8, 4));
    model.add_layer(std::make_shared<ReLU>());
    model.add_layer(std::make_shared<Linear>(4, 2));
    
    model.print_parameters();
    std::cout << "\n";
    
    // Создаем датасет: XOR (исключающее ИЛИ)
    std::vector<Tensor> inputs = {
        Tensor({1, 2}, {0.0f, 0.0f}),
        Tensor({1, 2}, {0.0f, 1.0f}),
        Tensor({1, 2}, {1.0f, 0.0f}),
        Tensor({1, 2}, {1.0f, 1.0f})
    };
    
    std::vector<Tensor> targets = {
        Tensor({1, 1}, {0.0f}),
        Tensor({1, 1}, {1.0f}),
        Tensor({1, 1}, {1.0f}),
        Tensor({1, 1}, {0.0f})
    };
    
    // Обучение с лучшими параметрами
    float learning_rate = 0.01f;  // Adam работает с меньшим LR
    int epochs = 5000;
    
    std::cout << "Training for " << epochs << " epochs with lr=" << learning_rate << "...\n\n";
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Проходим по всем примерам
        for (size_t i = 0; i < inputs.size(); ++i) {
            model.train(inputs[i], targets[i], learning_rate);
        }
        
        // Выводим прогресс каждые 1000 эпох
        if ((epoch + 1) % 1000 == 0) {
            std::cout << "\n--- Epoch " << epoch + 1 << " ---\n";
            float total_loss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                Tensor output = model.forward(inputs[i]);
                // Вычисляем loss для отображения
                CrossEntropyLoss loss_fn;
                float loss = loss_fn.forward(output, targets[i]);
                total_loss += loss;
                
                int prediction = (output[0] > output[1]) ? 0 : 1;
                std::cout << "Input: (" << inputs[i][0] << ", " << inputs[i][1] 
                          << ") -> Pred: " << prediction
                          << " (p0=" << output[0] << ", p1=" << output[1] << ")"
                          << " | Target: " << targets[i][0] << std::endl;
            }
            std::cout << "Average Loss: " << total_loss / 4.0f << std::endl;
            std::cout << std::endl;
        }
    }
    
    // Финальные предсказания
    std::cout << "\n=== Final Predictions ===\n";
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor output = model.forward(inputs[i]);
        int prediction = (output[0] > output[1]) ? 0 : 1;
        int target = (int)targets[i][0];
        correct += (prediction == target);
        
        std::cout << "Input: (" << inputs[i][0] << ", " << inputs[i][1] 
                  << ") -> Predicted: " << prediction 
                  << " (probs: " << output[0] << ", " << output[1] << ")"
                  << " | Target: " << target
                  << (prediction == target ? " ✓" : " ✗") << std::endl;
    }
    std::cout << "\nAccuracy: " << correct << "/4 (" << (correct * 100.0f / 4.0f) << "%)\n";
    system("pause");
    return 0;
}