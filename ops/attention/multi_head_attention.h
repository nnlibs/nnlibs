#pragma once

#include <ctime>

#include "self_attention_cpu.h"

class MultiHeadAttention : public Operator {
 public:
  MultiHeadAttention(int input_dim, int num_heads)
      : input_dim(input_dim), num_heads(num_heads) {
    head_dim = input_dim / num_heads;

    for (int i = 0; i < num_heads; ++i) {
      self_attentions.emplace_back(
          std::make_shared<SelfAttentionCPU>(head_dim, head_dim));
    }
    concat_out_weights =
        std::make_shared<Tensor>(std::vector<int>{input_dim, input_dim});

    // 随机初始化权重
    std::srand(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < concat_out_weights->Size(); ++i) {
      concat_out_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
  }

  MultiHeadAttention(int input_dim, int num_heads,
                     const std::vector<std::shared_ptr<Tensor>> query_weights,
                     const std::vector<std::shared_ptr<Tensor>> key_weights,
                     const std::vector<std::shared_ptr<Tensor>> value_weights,
                     const std::vector<std::shared_ptr<Tensor>> output_weights,
                     const std::shared_ptr<Tensor> concat_out_weights)
      : input_dim(input_dim),
        num_heads(num_heads),
        concat_out_weights(concat_out_weights) {
    head_dim = input_dim / num_heads;
    for (int i = 0; i < num_heads; ++i) {
      self_attentions.emplace_back(std::make_shared<SelfAttentionCPU>(
          input_dim, head_dim, query_weights[i], key_weights[i],
          value_weights[i], output_weights[i]));
    }
  }

  ~MultiHeadAttention() {}

 protected:
  std::shared_ptr<Tensor> concat_out_weights;  // 输出权重
  int num_heads;                               // 注意力头数
  int head_dim;                                // 每个头的维度
  int input_dim;                               // 输入维度

  std::shared_ptr<Tensor> grad_concat_out_weights;

  std::vector<std::shared_ptr<SelfAttention>> self_attentions;
};
