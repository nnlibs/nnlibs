#pragma once

#include "relu.h"

class ReluCPU : public Relu {
  public:
    ReluCPU(bool inplace = false);

    std::shared_ptr<Tensor>
    Forward(const std::shared_ptr<Tensor> input) override final;

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                     float learning_rate = 0.001f,
                                     float momentum = 0.0f) override final;
    std::shared_ptr<Tensor> Parameters() override final;

    void ZeroGrad() override final;

    ~ReluCPU();
};
