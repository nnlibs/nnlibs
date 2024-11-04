#pragma once

#include <iostream>
#include <random>

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

        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0);

        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights->data[i * in_features + j] =
                    distribution(generator) * 0.01;
            }
            bias->data[i] = 0.0;
        }
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
    std::shared_ptr<Tensor> weights; // shape: (out_features, in_features)
    std::shared_ptr<Tensor> bias;    // shape: (out_features)
    int in_features;
    int out_features;

    std::shared_ptr<Tensor> input_bakup;
};
