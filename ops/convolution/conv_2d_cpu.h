#pragma once

#include <iostream>

#include "conv_2d.h"

class Conv2dCPU : public Conv2d {
  public:
    Conv2dCPU(int in_channel, int out_channel, int kernel_h, int kernel_w,
              int stride, int padding, const std::shared_ptr<Tensor> weights,
              const std::shared_ptr<Tensor> bias);

    Conv2dCPU(int in_channel, int out_channel, int kernel_h, int kernel_w,
              int stride, int padding);

    Conv2dCPU(int in_channel, int out_channel, int kernel_size);

    std::shared_ptr<Tensor>
    Forward(const std::shared_ptr<Tensor> input) override final;

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> grad_output,
                                     float learning_rate) override final;

    std::shared_ptr<Tensor> Parameters() override final;

    void ZeroGrad() override final;

    ~Conv2dCPU();
};
