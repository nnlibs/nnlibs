#pragma once

#include "functional/functional.h"
#include <iostream>

#include "operator.h"

class Linear : public Operator {
  public:
    Linear(int in_features, int out_features,
           const std::shared_ptr<Tensor> weights,
           const std::shared_ptr<Tensor> bias)
        : in_features(in_features), out_features(out_features),
          weights(weights), bias(bias) {}

    Linear(int in_features, int out_features)
        : in_features(in_features), out_features(out_features) {
        weights = std::make_shared<Tensor>(
            std::vector<int>{out_features, in_features});
        bias = std::make_shared<Tensor>(std::vector<int>{out_features});
        weights_momentum = std::make_shared<Tensor>(
            std::vector<int>{out_features, in_features});
        bias_momentum =
            std::make_shared<Tensor>(std::vector<int>{out_features});

        F::kaiming_normal(weights, in_features);
        for (int i = 0; i < out_features; ++i) {
            bias->data[i] = 0.0f;
        }
    }

    std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> input) {
        std::cout << "Linear forward not implement" << std::endl;
    }

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                     float learning_rate, float momentum) {
        std::cout << "Linear backward not implement" << std::endl;
    }

    ~Linear() {}

  protected:
    std::shared_ptr<Tensor> weights; // shape: (out_features, in_features)
    std::shared_ptr<Tensor> bias;    // shape: (out_features)
    std::shared_ptr<Tensor>
        weights_momentum;                  // shape: (out_features, in_features)
    std::shared_ptr<Tensor> bias_momentum; // shape: (out_features)
    int in_features;
    int out_features;

    std::shared_ptr<Tensor> input_bakup;
};
