#pragma once

#include "relu.h"

class ReluCPU : public Relu {
 public:
  ReluCPU();

  std::shared_ptr<Tensor> Forward(
      const std::shared_ptr<Tensor> input) override final;

  std::shared_ptr<Tensor> Backward(
      const std::shared_ptr<Tensor> output) override final;

  // std::vector<float> params() { return {}; }
  // std::vector<float> grads() { return {}; }

  ~ReluCPU();
};