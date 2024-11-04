#include "cross_entropy.h"
#include <algorithm>
#include <cmath>

CrossEntropy::CrossEntropy() {}

float CrossEntropy::softmax(const std::shared_ptr<Tensor> logits, int index) {
    float max_logit =
        *std::max_element(logits->data.begin(), logits->data.end());
    float exp_sum = 0.0;
    for (int i = 0; i < logits->Size(); ++i) {
        exp_sum +=
            std::exp(logits->data[i] - max_logit); // 减去最大值以提高数值稳定性
    }
    return exp(logits->data[index] - max_logit) / exp_sum; // 计算 softmax
}

float CrossEntropy::Forward(const std::shared_ptr<Tensor> logits,
                            const std::shared_ptr<Tensor> target) {
    assert(logits->Size() == target->Size() &&
           "logits and target must have the same size.");

    float loss = 0.0;
    for (int i = 0; i < logits->Size(); ++i) {
        double prob = softmax(logits, i);
        loss -=
            target->data[i] * std::log(prob + 1e-12); // 加上小常数防止 log(0)
    }
    return loss / logits->Size(); // 返回平均损失
}

std::shared_ptr<Tensor>
CrossEntropy::Backward(const std::shared_ptr<Tensor> logits,
                       const std::shared_ptr<Tensor> target) {
    assert(logits->Size() == target->Size() &&
           "Logits and target must have the same size.");

    auto grad = std::make_shared<Tensor>(logits->shape);
    for (int i = 0; i < logits->Size(); ++i) {
        double prob = softmax(logits, i);
        grad->data[i] = prob - target->data[i]; // 计算梯度
    }
    return grad; // 返回梯度
}

std::shared_ptr<Tensor> Parameters() { return nullptr; }

CrossEntropy::~CrossEntropy() {}
