#pragma once

#include "layer_norm.h"

class LayerNormCPU : public LayerNorm {
  public:
    LayerNormCPU(const std::vector<int> &norm_shape);

    std::shared_ptr<Tensor>
    Forward(std::shared_ptr<Tensor> input) override final;

    std::shared_ptr<Tensor> Backward(std::shared_ptr<Tensor> input,
                                     float lr) override final;
    std::shared_ptr<Tensor> Parameters() override final;

    void ZeroGrad() override final;

    ~LayerNormCPU();
};
