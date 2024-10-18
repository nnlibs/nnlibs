#include "linear_cpu.h"

LinearCPU::LinearCPU(int in_features, int out_features,
                     const std::shared_ptr<Tensor> weight,
                     const std::shared_ptr<Tensor> bias)
    : Linear(in_features, out_features, weight, bias) {}

std::shared_ptr<Tensor> LinearCPU::Forward(
    const std::shared_ptr<Tensor> input) {
  // or (1,out_features)
  std::shared_ptr<Tensor> output =
      std::make_shared<Tensor>(std::vector<int>{1, out_features});
  for (int i = 0; i < out_features; i++) {
    output->data[i] = bias->data[i];
    for (int j = 0; j < in_features; j++) {
      output->data[i] += input->data[j] * weights->data[i * in_features + j];
      // TODO: gradient need add?
      // weights->grad[i * in_features + j] += input->data[j] * output->grad[i];
    }
  }
  return output;
}

std::shared_ptr<Tensor> LinearCPU::Backward(
    const std::shared_ptr<Tensor> output) {
  // return std::make_shared<Tensor>(in_features);
  return nullptr;
}

void LinearCPU::UpdateParams(float learning_rate) {
  for (int i = 0; i < out_features; i++) {
    for (int j = 0; j < in_features; j++) {
      weights->data[i * in_features + j] -=
          learning_rate * weights->grad[i * in_features + j];
    }
    bias->data[i] -= learning_rate * bias->grad[i];
  }
}

LinearCPU::~LinearCPU() {}