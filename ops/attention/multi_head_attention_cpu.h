#pragma once

#include "multi_head_attention.h"

class MultiHeadAttentionCPU : public MultiHeadAttention {
  public:
    MultiHeadAttentionCPU(int input_dim, int num_heads);

    MultiHeadAttentionCPU(
        int input_dim, int num_heads,
        const std::vector<std::shared_ptr<Tensor>> query_weights,
        const std::vector<std::shared_ptr<Tensor>> key_weights,
        const std::vector<std::shared_ptr<Tensor>> value_weights,
        const std::vector<std::shared_ptr<Tensor>> output_weights,
        const std::shared_ptr<Tensor> concat_out_weights);

    std::shared_ptr<Tensor>
    Forward(const std::shared_ptr<Tensor> input) override final;

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                     float learning_rate = 0.001f,
                                     float momentum = 0.0f) override final;

    std::shared_ptr<Tensor> Parameters() override final;

    void ZeroGrad() override final;

    ~MultiHeadAttentionCPU();
};
