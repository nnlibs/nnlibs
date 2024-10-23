#pragma once

#include "self_attention.h"

class SelfAttentionCPU : public SelfAttention {
 public:
  SelfAttentionCPU(int input_dim);

  SelfAttentionCPU(int input_dim, const std::shared_ptr<Tensor> query_weights,
                   const std::shared_ptr<Tensor> key_weights,
                   const std::shared_ptr<Tensor> value_weights,
                   const std::shared_ptr<Tensor> output_weights);

  std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input) override final;

  std::shared_ptr<Tensor> Backward(std::shared_ptr<Tensor> grad_output,
                                   float lr) override final;

  ~SelfAttentionCPU();
};
