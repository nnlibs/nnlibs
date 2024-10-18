#pragma once

#include "operator.h"

class Sigmoid : public Operator {
 public:
  virtual std::shared_ptr<Tensor> Forward(
      const std::shared_ptr<Tensor> input) = 0;

  virtual std::shared_ptr<Tensor> Backward(
      const std::shared_ptr<Tensor> output) = 0;

  // std::vector<float> params() { return {}; }
  // std::vector<float> grads() { return {}; }
};