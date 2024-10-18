#pragma once

#include "sigmoid_cpu.h"

#include <cmath>

SigmoidCPU::SigmoidCPU() : Sigmoid() {}

std::shared_ptr<Tensor> SigmoidCPU::Forward(
    const std::shared_ptr<Tensor> input) {
  // in-place perf
  for (int i = 0; i < input->Size(); i++) {
    input->data[i] = 1.0 / (1.0 + exp(-input->data[i]));
  }
  return input;
}

// f′(x) = f(x)(1 − f(x))​
std::shared_ptr<Tensor> SigmoidCPU::Backward(
    const std::shared_ptr<Tensor> input) {
  for (int i = 0; i < input->Size(); i++) {
    input->data[i] = input->data[i] * (1 - input->data[i]);
  }
  return nullptr;
}

SigmoidCPU::~SigmoidCPU() {}