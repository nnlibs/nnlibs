#include "multi_head_attention_cpu.h"

#include <cmath>
#include <iostream>

MultiHeadAttentionCPU::MultiHeadAttentionCPU(int input_dim, int num_head)
    : MultiHeadAttention(input_dim, num_head) {}

MultiHeadAttentionCPU::MultiHeadAttentionCPU(
    int input_dim, int num_head,
    const std::vector<std::shared_ptr<Tensor>> query_weights,
    const std::vector<std::shared_ptr<Tensor>> key_weights,
    const std::vector<std::shared_ptr<Tensor>> value_weights,
    const std::vector<std::shared_ptr<Tensor>> output_weights,
    const std::shared_ptr<Tensor> concat_out_weights)
    : MultiHeadAttention(input_dim, num_head, query_weights, key_weights,
                         value_weights, output_weights, concat_out_weights) {}

std::shared_ptr<Tensor>
MultiHeadAttentionCPU::Forward(std::shared_ptr<Tensor> input) {
    int seq_len = input->Size() / input_dim;
    std::shared_ptr<Tensor> concat_outs =
        std::make_shared<Tensor>(std::vector<int>{seq_len, input_dim});
    auto &concat_outs_data = *concat_outs;

    for (int i = 0; i < num_heads; ++i) {
        std::cout << "head " << i << " forward ...\n";
        auto self_atte_out = self_attentions[i]->Forward(input); // 单个头的输出
        auto &self_atte_data = *self_atte_out;
        // 将单个头的输出拼接起来
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < head_dim; ++k) {
                concat_outs_data({j, i * head_dim + k}) =
                    self_atte_data({j, k});
            }
        }
    }

    // 最后通过线性变换输出，可以让输出的每个词向量维度和输入保持一致
    std::shared_ptr<Tensor> final_output =
        std::make_shared<Tensor>(std::vector<int>{seq_len, input_dim});
    auto &fo_data = *final_output;
    auto &ow_data = *concat_out_weights;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            fo_data({i, j}) = 0.0f;
            for (int k = 0; k < input_dim; ++k) {
                fo_data({i, j}) += concat_outs_data({i, k}) * ow_data({k, j});
            }
        }
    }
    return final_output;
}

std::shared_ptr<Tensor>
MultiHeadAttentionCPU::Backward(std::shared_ptr<Tensor> grad_output, float lr) {
}

std::shared_ptr<Tensor> MultiHeadAttentionCPU::Parameters() { return nullptr; }

void MultiHeadAttentionCPU::ZeroGrad() {}

MultiHeadAttentionCPU ::~MultiHeadAttentionCPU() {}
