#pragma once

#include "operator.h"

class Relu : public Operator {
 public:
  // std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> input);

  // std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output);

  // std::vector<float> params() { return {}; }
  // std::vector<float> grads() { return {}; }

 protected:
  std::shared_ptr<Tensor> grad = nullptr;
};