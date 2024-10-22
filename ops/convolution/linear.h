#pragma once

#include <iostream>

#include "operator.h"

class Linear : public Operator {
 public:
  Linear(int in_features, int out_features,
         const std::shared_ptr<Tensor> weights,
         const std::shared_ptr<Tensor> bias)
      : in_features(in_features),
        out_features(out_features),
        weights(weights),
        bias(bias) {
    grad_w = std::make_shared<Tensor>(weights->shape);
    grad_b = std::make_shared<Tensor>(bias->shape);
  }

  Linear(int in_features, int out_features)
      : in_features(in_features), out_features(out_features) {
    weights =
        std::make_shared<Tensor>(std::vector<int>{out_features, in_features});
    bias = std::make_shared<Tensor>(std::vector<int>{out_features});
  }

  std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> input) {
    std::cout << "Linear forward not implement" << std::endl;
  }

  std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                   float learning_rate) {
    std::cout << "Linear backward not implement" << std::endl;
  }

  ~Linear() {}

 protected:
  std::shared_ptr<Tensor> weights;  // shape: (out_features, in_features)
  std::shared_ptr<Tensor> bias;
  int in_features;
  int out_features;

  std::shared_ptr<Tensor> grad_w;
  std::shared_ptr<Tensor> grad_b;
};