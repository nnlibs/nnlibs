#include "linear_cpu.h"

LinearCPU::LinearCPU(int in_features, int out_features,
                     const std::shared_ptr<Tensor> weight,
                     const std::shared_ptr<Tensor> bias)
    : Linear(in_features, out_features, weight, bias) {}

LinearCPU::LinearCPU(int in_features, int out_features)
    : Linear(in_features, out_features) {}

std::shared_ptr<Tensor>
LinearCPU::Forward(const std::shared_ptr<Tensor> input) {
    // @LIMIT: only support one dims input
    if (input->Size() != input->shape.back()) {
        std::cout << "input dims is not one, Fatal" << std::endl;
        return nullptr;
    }
    // save input
    input_bakup = input;

    std::shared_ptr<Tensor> output =
        std::make_shared<Tensor>(std::vector<int>{1, out_features});
    for (int i = 0; i < out_features; i++) {
        output->data[i] = bias->data[i];
        for (int j = 0; j < in_features; j++) {
            output->data[i] +=
                input->data[j] * weights->data[i * in_features + j];
        }
    }
    return output;
}

std::shared_ptr<Tensor>
LinearCPU::Backward(const std::shared_ptr<Tensor> grad_output,
                    float learning_rate, float momentum) {
    std::shared_ptr<Tensor> diff =
        std::make_shared<Tensor>(std::vector<int>{in_features});
    std::shared_ptr<Tensor> grad_weights =
        std::make_shared<Tensor>(std::vector<int>{out_features, in_features});
    std::shared_ptr<Tensor> grad_bias =
        std::make_shared<Tensor>(std::vector<int>{out_features});

    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < in_features; j++) {
            grad_weights->data[i * in_features + j] =
                grad_output->data[i] * input_bakup->data[j]; // weight gradient
            diff->data[j] +=
                grad_output->data[i] *
                weights->data[i * in_features + j]; // input gradient
        }
        grad_bias->data[i] += grad_output->data[i]; // bias gradient
    }

    // update weight and bias features
    // update weight/bias
    for (int i = 0; i < grad_weights->Size(); i++) {
        weights_momentum->data[i] = momentum * weights_momentum->data[i] +
                                    (1 - momentum) * grad_weights->data[i];
        weights->data[i] -= learning_rate * weights_momentum->data[i];
    }
    for (int i = 0; i < grad_bias->Size(); i++) {
        bias_momentum->data[i] = momentum * bias_momentum->data[i] +
                                 (1 - momentum) * grad_bias->data[i];
        bias->data[i] -= learning_rate * bias_momentum->data[i];
    }

    return diff;
}
std::shared_ptr<Tensor> LinearCPU::Parameters() { return nullptr; }

void LinearCPU::ZeroGrad() {}

LinearCPU::~LinearCPU() {}
