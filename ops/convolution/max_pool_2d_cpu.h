#pragma once

#include "max_pool_2d.h"

class MaxPool2dCpu : public MaxPool2d {
  public:
    MaxPool2dCpu(int kernel_size, int stride = -1);

    std::shared_ptr<Tensor>
    Forward(const std::shared_ptr<Tensor> input) override final;

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> grad_output,
                                     float learning_rate = 0.001f,
                                     float momentum = 0.0f) override final;
    std::shared_ptr<Tensor> Parameters() override final;

    void ZeroGrad() override final;

    ~MaxPool2dCpu();
};
