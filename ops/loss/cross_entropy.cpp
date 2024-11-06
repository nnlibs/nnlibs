#include "cross_entropy.h"
#include <algorithm>
#include <cmath>

CrossEntropy::CrossEntropy() {}

std::shared_ptr<Tensor>
CrossEntropy::softmax(const std::shared_ptr<Tensor> logits) {
    float max_logit =
        *std::max_element(logits->data.begin(), logits->data.end());
    auto exp_vals = std::make_shared<Tensor>(logits->shape);
    float sum_exp = 0.0;

    for (size_t i = 0; i < logits->Size(); ++i) {
        exp_vals->data[i] = std::exp(logits->data[i] - max_logit); // 防止溢出
        sum_exp += exp_vals->data[i];
    }

    for (size_t i = 0; i < logits->Size(); ++i) {
        exp_vals->data[i] /= sum_exp;
    }

    return exp_vals;
}

float CrossEntropy::Forward(const std::shared_ptr<Tensor> logits,
                            const std::shared_ptr<Tensor> label) {
    auto probs = softmax(logits);
    return -std::log(probs->data[label->data[0]]); // 计算交叉熵
}

std::shared_ptr<Tensor>
CrossEntropy::Backward(const std::shared_ptr<Tensor> logits,
                       const std::shared_ptr<Tensor> target) {
    auto grad = softmax(logits);
    grad->data[target->data[0]] -= 1.0;
    return grad;
}

std::shared_ptr<Tensor> Parameters() { return nullptr; }

CrossEntropy::~CrossEntropy() {}
