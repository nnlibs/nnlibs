#include "linear_cpu.h"
#include <immintrin.h>
#include <x86intrin.h>

LinearCPU::LinearCPU(int in_features, int out_features,
                     const std::shared_ptr<Tensor> weight,
                     const std::shared_ptr<Tensor> bias)
    : Linear(in_features, out_features, weight, bias) {}

LinearCPU::LinearCPU(int in_features, int out_features)
    : Linear(in_features, out_features) {}
std::shared_ptr<Tensor>
LinearCPU::Forward(const std::shared_ptr<Tensor> input) {
    if (input->Size() != input->shape.back()) {
        std::cout << "the last dim value is not equal to input size"
                  << std::endl;
        return nullptr;
    }
    input_bakup = input;

    std::shared_ptr<Tensor> output =
        std::make_shared<Tensor>(std::vector<int>{1, out_features});
    auto &bias_mut = *bias;
    auto &output_mut = *output;
    auto &input_mut = *input;

    for (int i = 0; i < out_features; i++) {
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        for (; j <= in_features - 8; j += 8) {
            __m256 w = _mm256_loadu_ps(&weights->data[i * in_features + j]);
            __m256 x = _mm256_loadu_ps(&input->data[j]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(w, x));
        }
        alignas(32) float sumArray[8];
        _mm256_store_ps(sumArray, sum);
        float result = sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3] +
                       sumArray[4] + sumArray[5] + sumArray[6] + sumArray[7];
        for (; j < in_features; j++) {
            result += weights->data[i * in_features + j] * input_mut[j];
        }
        output_mut[i] = result + bias_mut[i];
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
        int j = 0;
        for (; j <= in_features - 8; j += 8) {
            __m256 grad_m = _mm256_set1_ps(grad_output->data[i]);
            __m256 input_m = _mm256_loadu_ps(&input_bakup->data[j]);
            __m256 weight_m =
                _mm256_loadu_ps(&weights->data[i * in_features + j]);
            _mm256_storeu_ps(&grad_weights->data[i * in_features + j],
                             _mm256_mul_ps(grad_m, input_m));
            _mm256_storeu_ps(&diff->data[j],
                             _mm256_add_ps(_mm256_loadu_ps(&diff->data[j]),
                                           _mm256_mul_ps(grad_m, weight_m)));
        }
        for (; j < in_features; j++) {
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

// std::shared_ptr<Tensor>
// LinearCPU::Forward(const std::shared_ptr<Tensor> input) {
//     // @LIMIT: only support one dims input
//     if (input->Size() != input->shape.back()) {
//         std::cout << "input dims is not one, Fatal" << std::endl;
//         return nullptr;
//     }
//     // save input
//     input_bakup = input;

//     std::shared_ptr<Tensor> output =
//         std::make_shared<Tensor>(std::vector<int>{1, out_features});
//     for (int i = 0; i < out_features; i++) {
//         output->data[i] = bias->data[i];
//         for (int j = 0; j < in_features; j++) {
//             output->data[i] +=
//                 input->data[j] * weights->data[i * in_features + j];
//         }
//     }
//     return output;
// }

// std::shared_ptr<Tensor>
// LinearCPU::Backward(const std::shared_ptr<Tensor> grad_output,
//                     float learning_rate, float momentum) {
//     std::shared_ptr<Tensor> diff =
//         std::make_shared<Tensor>(std::vector<int>{in_features});
//     std::shared_ptr<Tensor> grad_weights =
//         std::make_shared<Tensor>(std::vector<int>{out_features,
//         in_features});
//     std::shared_ptr<Tensor> grad_bias =
//         std::make_shared<Tensor>(std::vector<int>{out_features});

//     for (int i = 0; i < out_features; i++) {
//         for (int j = 0; j < in_features; j++) {
//             grad_weights->data[i * in_features + j] =
//                 grad_output->data[i] * input_bakup->data[j]; // weight
//                 gradient
//             diff->data[j] +=
//                 grad_output->data[i] *
//                 weights->data[i * in_features + j]; // input gradient
//         }
//         grad_bias->data[i] += grad_output->data[i]; // bias gradient
//     }

//     // update weight and bias features
//     // update weight/bias
//     for (int i = 0; i < grad_weights->Size(); i++) {
//         weights_momentum->data[i] = momentum * weights_momentum->data[i] +
//                                     (1 - momentum) * grad_weights->data[i];
//         weights->data[i] -= learning_rate * weights_momentum->data[i];
//     }
//     for (int i = 0; i < grad_bias->Size(); i++) {
//         bias_momentum->data[i] = momentum * bias_momentum->data[i] +
//                                  (1 - momentum) * grad_bias->data[i];
//         bias->data[i] -= learning_rate * bias_momentum->data[i];
//     }

//     return diff;
// }

std::shared_ptr<Tensor> LinearCPU::Parameters() { return nullptr; }

void LinearCPU::ZeroGrad() {}

LinearCPU::~LinearCPU() {}
