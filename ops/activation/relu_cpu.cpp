#include "relu_cpu.h"
#include <x86intrin.h>

ReluCPU::ReluCPU(bool inplace) : Relu(inplace) {}

std::shared_ptr<Tensor> ReluCPU::Forward(const std::shared_ptr<Tensor> input) {
    grad = std::make_shared<Tensor>(input->shape);
    std::shared_ptr<Tensor> output;
    if (inplace) {
        output = input;
    } else {
        output = std::make_shared<Tensor>(input->shape);
    }
    int i = 0;
    auto size = input->Size();
    // SIMD acceleration
    // why not use OpenMP?: the time of thread schedule is longer, so the
    // speedup is not obvious
    __m256 zero = _mm256_setzero_ps();
    for (; i <= size - 8; i += 8) {
        __m256 input_vec = _mm256_loadu_ps(&input->data[i]);
        __m256 relu_result = _mm256_max_ps(input_vec, zero);
        _mm256_storeu_ps(&output->data[i], relu_result);

        // compare if input > 0
        __m256 grad_result = _mm256_cmp_ps(input_vec, zero, _CMP_GT_OS);
        // set gradient to 1 or 0
        grad_result = _mm256_blendv_ps(zero, _mm256_set1_ps(1.0f), grad_result);

        _mm256_storeu_ps(&grad->data[i], grad_result);
    }
    for (; i < size; i++) {
        output->data[i] = std::max(0.0f, input->data[i]);
        grad->data[i] = input->data[i] > 0 ? 1 : 0;
    }
    return output;
}

std::shared_ptr<Tensor> ReluCPU::Backward(const std::shared_ptr<Tensor> input,
                                          float learning_rate, float momentum) {
    // learning_rate and momentum only used for trained layer
    std::shared_ptr<Tensor> input_grad = std::make_shared<Tensor>(input->shape);
    int i = 0;
    auto size = input->Size();
    for (; i <= size - 8; i += 8) {
        _mm256_storeu_ps(&input_grad->data[i],
                         _mm256_mul_ps(_mm256_loadu_ps(&grad->data[i]),
                                       _mm256_loadu_ps(&input->data[i])));
    }
    for (; i < size; i++) {
        input_grad->data[i] = input->data[i] * grad->data[i];
    }
    return input_grad;
}

void ReluCPU::ZeroGrad() { grad.reset(); }

std::shared_ptr<Tensor> ReluCPU::Parameters() { return nullptr; }

ReluCPU::~ReluCPU() {}
