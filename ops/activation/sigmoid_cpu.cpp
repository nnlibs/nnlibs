#include "sigmoid_cpu.h"

#include <cmath>

SigmoidCPU::SigmoidCPU() : Sigmoid() {}

std::shared_ptr<Tensor> SigmoidCPU::Forward(
    const std::shared_ptr<Tensor> input) {
  // in-place perf
  grad.reset(new Tensor(input->shape));
  for (int i = 0; i < input->Size(); i++) {
    input->data[i] = 1.0 / (1.0 + exp(-input->data[i]));
    // f'(x) = f(x)(1 − f(x))​
    grad->data[i] = input->data[i] * (1 - input->data[i]);
  }
  return input;
}

std::shared_ptr<Tensor> SigmoidCPU::Backward(
    const std::shared_ptr<Tensor> input, float learning_rate) {
  std::shared_ptr<Tensor> diff = std::make_shared<Tensor>(input->shape);
  // for (int i = 0; i < input->Size(); i++) {
  //   diff->data[i] = 0;
  // }
  for (int i = 0; i < input->Size(); i++) {
    diff->data[i] = input->data[i] * grad->data[i];
  }
  return diff;
}

SigmoidCPU::~SigmoidCPU() {}