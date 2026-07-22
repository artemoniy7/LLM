#include "include/tensor.hpp"
#include <iostream>

int main() {
    std::cout << "=== Testing Tensor Class ===\n\n";
    
    // 1. Создание и инициализация
    std::cout << "1. Creating tensors...\n";
    Tensor a({2, 3});
    a.random_uniform(-1.0f, 1.0f);
    std::cout << "a: ";
    a.print();
    
    Tensor b({2, 3});
    b.fill(2.0f);
    std::cout << "b: ";
    b.print();
    
    // 2. Сложение
    std::cout << "\n2. Addition: a + b\n";
    Tensor c = a + b;
    c.print();
    
    // 3. Умножение на скаляр
    std::cout << "\n3. Scalar multiplication: a * 0.5\n";
    Tensor d = a * 0.5f;
    d.print();
    
    // 4. Матричное умножение
    std::cout << "\n4. Matrix multiplication...\n";
    Tensor e({2, 3});
    e.fill(1.0f);
    std::cout << "e (2x3):\n";
    e.print();
    
    Tensor f({3, 2});
    f.fill(2.0f);
    std::cout << "f (3x2):\n";
    f.print();
    
    Tensor g = e.matmul(f);
    std::cout << "e.matmul(f) (2x2):\n";
    g.print();
    system("pause");
    return 0;
}