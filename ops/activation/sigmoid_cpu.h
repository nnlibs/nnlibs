#pragma once

#include "sigmoid.h"

class SigmoidCPU : public Sigmoid {
  public:
    SigmoidCPU();

    std::shared_ptr<Tensor>
    Forward(const std::shared_ptr<Tensor> input) override final;

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                     float learning_rate) override final;
    std::shared_ptr<Tensor> Parameters() override final;

    void ZeroGrad() override final;
    ~SigmoidCPU();
};
