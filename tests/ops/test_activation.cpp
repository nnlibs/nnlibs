#include <iostream>

#include "ops/activation/relu_cpu.h"

int main() {
    std::shared_ptr<Tensor> t_ptr =
        std::make_shared<Tensor>(std::vector<int>{2, 3});
    auto &t = *t_ptr;
    t({0, 0}) = 1;
    t({0, 1}) = 2;
    t({0, 2}) = -3;
    t({1, 0}) = 4;
    t({1, 1}) = -5;
    t({1, 2}) = 6;

    Relu *r = new ReluCPU();
    auto o = r->Forward(t_ptr);
    std::cout << "output: " << o << std::endl;

    auto grad_output = std::make_shared<Tensor>(std::vector<int>{2, 3});
    for (int i = 0; i < grad_output->Size(); ++i) {
        grad_output->data[i] = 1;
    }
    auto grad_input = r->Backward(grad_output, 0.001, 0);
    std::cout << "grad: " << grad_input << std::endl;

    return 0;
}
