#pragma once

#include <memory>

#include "common/tensor.h"

class Operator {
 public:
  Operator() {}

  virtual std::shared_ptr<Tensor> Forward(
      const std::shared_ptr<Tensor> input) = 0;

  virtual std::shared_ptr<Tensor> Backward(
      const std::shared_ptr<Tensor> output) = 0;

  virtual void UpdateParams(float learning_rate) {}

  virtual ~Operator() {}
};