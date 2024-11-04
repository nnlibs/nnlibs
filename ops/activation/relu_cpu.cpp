#include "relu_cpu.h"

ReluCPU::ReluCPU() : Relu() {}

std::shared_ptr<Tensor> ReluCPU::Forward(const std::shared_ptr<Tensor> input) {
    grad = std::make_shared<Tensor>(input->shape);
    for (int i = 0; i < input->Size(); i++) {
        input->data[i] = std::max(0.0f, input->data[i]);
        grad->data[i] = input->data[i] > 0 ? 1 : 0;
    }
    return input;
}

std::shared_ptr<Tensor> ReluCPU::Backward(const std::shared_ptr<Tensor> input,
                                          float learning_rate) {
    std::shared_ptr<Tensor> diff = std::make_shared<Tensor>(input->shape);
    for (int i = 0; i < diff->Size(); i++) {
        diff->data[i] = input->data[i] * grad->data[i];
    }
    return diff;
}

void ReluCPU::ZeroGrad() { grad.reset(); }

std::shared_ptr<Tensor> ReluCPU::Parameters() { return nullptr; }

ReluCPU::~ReluCPU() {}
