#pragma once

#include "linear.h"

class LinearCPU : public Linear {
 public:
  LinearCPU(int in_features, int out_features,
            const std::shared_ptr<Tensor> weights,
            const std::shared_ptr<Tensor> bias);

  std::shared_ptr<Tensor> Forward(
      const std::shared_ptr<Tensor> input) override final;

  std::shared_ptr<Tensor> Backward(
      const std::shared_ptr<Tensor> output) override final;

  void UpdateParams(float learning_rate) override final;

  ~LinearCPU();
};