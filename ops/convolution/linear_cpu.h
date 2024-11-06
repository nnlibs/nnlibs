#pragma once

#include "linear.h"

class LinearCPU : public Linear {
  public:
    LinearCPU(int in_features, int out_features,
              const std::shared_ptr<Tensor> weights,
              const std::shared_ptr<Tensor> bias);

    LinearCPU(int in_features, int out_features);

    std::shared_ptr<Tensor>
    Forward(const std::shared_ptr<Tensor> input) override final;

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> grad_output,
                                     float learning_rate = 0.001f,
                                     float momentum = 0.0f) override final;
    std::shared_ptr<Tensor> Parameters() override final;

    void ZeroGrad() override final;

    ~LinearCPU();
};
