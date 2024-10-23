#include "self_attention_cpu.h"

#include <cmath>
#include <iostream>

SelfAttentionCPU::SelfAttentionCPU(int input_dim) : SelfAttention(input_dim) {}

SelfAttentionCPU::SelfAttentionCPU(int input_dim,
                                   const std::shared_ptr<Tensor> query_weights,
                                   const std::shared_ptr<Tensor> key_weights,
                                   const std::shared_ptr<Tensor> value_weights,
                                   const std::shared_ptr<Tensor> output_weights)
    : SelfAttention(input_dim, query_weights, key_weights, value_weights,
                    output_weights) {}

std::shared_ptr<Tensor> SelfAttentionCPU::Forward(
    std::shared_ptr<Tensor> input) {
  // input shape: (n, d)  n：输入词个数， d: 词向量dim
  // weigit_q shape: (d, d_k)
  // weight_k shape: (d, d_k)
  // weight_v shape: (d, d_v)

  // 1. q = input * weight_q, shape: (n, d_k)
  // 2. k = input * weight_k, shape: (n, d_k)
  // 3. v = input * weight_v, shape: (n, d_v)
  // 4. attention_scores = softmax[q * k^T / sqrt(d_k)], shape: (n, n)
  // 5. output = attention_scores * v, shape: (n, d_v)
  // 6. final_out = output * weight_o, shape: (n, d)

  // 输入尺寸
  int seq_len = input->Size() / input_dim;

  // 计算Q, K, V
  std::shared_ptr<Tensor> Q =
      std::make_shared<Tensor>(std::vector<int>{input_dim, input_dim});
  auto& q_data = *Q;
  auto& qw_data = *query_weights;
  std::shared_ptr<Tensor> K =
      std::make_shared<Tensor>(std::vector<int>{input_dim, input_dim});
  auto& k_data = *K;
  auto& kw_data = *key_weights;
  std::shared_ptr<Tensor> V =
      std::make_shared<Tensor>(std::vector<int>{input_dim, input_dim});
  auto& v_data = *V;
  auto& vw_data = *value_weights;

  auto& in_data = *input;
  // Q = input * query_weights
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      q_data({i, j}) = 0.0f;
      for (int k = 0; k < input_dim; ++k) {
        q_data({i, j}) += in_data({i, k}) * qw_data({j, k});
      }
    }
  }
  // K = input * key_weights
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      k_data({i, j}) = 0.0f;
      for (int k = 0; k < input_dim; ++k) {
        k_data({i, j}) += in_data({i, k}) * kw_data({j, k});
      }
    }
  }

  // V = input * value_weights
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      v_data({i, j}) = 0.0f;
      for (int k = 0; k < input_dim; ++k) {
        v_data({i, j}) += in_data({i, k}) * vw_data({j, k});
      }
    }
  }

  // 计算注意力得分
  std::shared_ptr<Tensor> attention_scores =
      std::make_shared<Tensor>(std::vector<int>{seq_len, seq_len});
  auto& at_score = *attention_scores;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      float dot_product = 0.0f;
      // q * k^T
      for (int d = 0; d < input_dim; ++d) {
        dot_product += q_data({i, d}) * k_data({j, d});
      }
      // / sqrt(d_k)
      at_score({i, j}) = dot_product / std::sqrt(static_cast<float>(input_dim));
      // std::cout << at_score({i, j}) << " " << dot_product << ", ";
    }
    // std::cout << std::endl;
  }

  // score Softmax
  std::shared_ptr<Tensor> attention_weights =
      std::make_shared<Tensor>(std::vector<int>{seq_len, seq_len});
  auto& aw_data = *attention_weights;
  for (int i = 0; i < seq_len; ++i) {
    // float max_score = attention_scores->data[i * seq_len];
    // for (int j = 1; j < seq_len; ++j) {
    //   max_score = std::max(max_score, attention_scores->data[i * seq_len +
    //   j]);
    // }
    float max_score = 0.0f;
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
      aw_data({i, j}) = std::exp(aw_data({i, j}) - max_score);
      sum_exp += aw_data({i, j});
    }
    for (int j = 0; j < seq_len; ++j) {
      aw_data({i, j}) /= sum_exp;
      std::cout << aw_data({i, j}) << " ";
    }
    std::cout << std::endl;
  }

  // 计算加权值
  std::shared_ptr<Tensor> output =
      std::make_shared<Tensor>(std::vector<int>{seq_len, input_dim});
  auto& out_data = *output;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      for (int d = 0; d < input_dim; ++d) {
        out_data({i, d}) = 0.0f;
      }
      for (int d = 0; d < input_dim; ++d) {
        out_data({i, d}) += aw_data({i, j}) * v_data({j, d});
      }
    }
  }

  // 最后通过线性变换输出，可以让输出的每个词向量维度和输入保持一致
  std::shared_ptr<Tensor> final_output =
      std::make_shared<Tensor>(std::vector<int>{seq_len, input_dim});
  auto& fo_data = *final_output;
  auto& ow_data = *output_weights;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      fo_data({i, j}) = 0.0f;
      for (int k = 0; k < input_dim; ++k) {
        fo_data({i, j}) += out_data({i, k}) * ow_data({k, j});
      }
    }
  }
  return final_output;
}

std::shared_ptr<Tensor> SelfAttentionCPU::Backward(
    std::shared_ptr<Tensor> grad_output, float lr) {}

SelfAttentionCPU ::~SelfAttentionCPU() {}
