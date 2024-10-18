#pragma once

#include "relu_cpu.h"

ReluCPU::ReluCPU() : Relu() {}

std::shared_ptr<Tensor> ReluCPU::Forward(const std::shared_ptr<Tensor> input) {
  // in-place perf
  for (int i = 0; i < input->Size(); i++) {
    input->data[i] = std::max(0.0f, input->data[i]);
  }
  return input;
}

std::shared_ptr<Tensor> ReluCPU::Backward(
    const std::shared_ptr<Tensor> output) {
  std::shared_ptr<Tensor> diff = std::make_shared<Tensor>(output->shape);
  for (int i = 0; i < diff->Size(); i++) {
    diff->data[i] = output->data[i] > 0 ? 1 : 0;  // * diff
  }
  return diff;
}

ReluCPU::~ReluCPU() {}