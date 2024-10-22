#pragma once

#include "ops/ops.h"

class Network {
 public:
  Network(bool use_gpu = false) {}

  virtual std::shared_ptr<Tensor> Forward(
      const std::shared_ptr<Tensor> input) = 0;

  virtual void Backward(const std::shared_ptr<Tensor> loss, float lr) = 0;

  virtual ~Network() {}

 protected:
  float learn_rate = 0.001;
};
