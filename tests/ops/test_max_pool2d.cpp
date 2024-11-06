#include "convolution/max_pool_2d_cpu.h"
#include <iostream>
#include <memory>
int main() {
    MaxPool2dCpu ce(2);

    auto input = std::make_shared<Tensor>(std::vector<int>{1, 1, 4, 4});

    for (int i = 0; i < 16; i++) {
        input->data[i] = i + 1;
    }

    std::cout << "input: " << input << std::endl;

    auto output = ce.Forward(input);
    std::cout << "output: " << output << std::endl;

    auto output_grad = std::make_shared<Tensor>(std::vector<int>{1, 1, 2, 2});
    for (int i = 0; i < 4; i++) {
        output_grad->data[i] = i + 1;
    }

    auto grad = ce.Backward(output_grad);
    std::cout << "Gradient: " << grad << std::endl;
    return 0;
}
