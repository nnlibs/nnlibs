#include "loss/cross_entropy.h"
#include <iostream>
#include <memory>
int main() {
    CrossEntropy ce;

    auto logits = std::make_shared<Tensor>(std::vector<int>{1, 3});
    auto target = std::make_shared<Tensor>(std::vector<int>{1, 3});

    logits->data = {2.0f, 1.0f, 0.1f};
    target->data = {1.0f, 0.0f, 0.0f};

    auto loss = ce.Forward(logits, target);
    std::cout << "Loss: " << loss << std::endl;

    auto grad = ce.Backward(logits, target);
    std::cout << "Gradient: " << grad << std::endl;
    return 0;
}
