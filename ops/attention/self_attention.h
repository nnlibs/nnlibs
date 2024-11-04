#pragma once

#include <cstdlib>
#include <ctime>

#include "operator.h"

class SelfAttention : public Operator {
  public:
    SelfAttention(int input_dim, int head_dim)
        : input_dim(input_dim), head_dim(head_dim) {
        // default(single head): head_dim = input_dim
        query_weights =
            std::make_shared<Tensor>(std::vector<int>{input_dim, head_dim});
        key_weights =
            std::make_shared<Tensor>(std::vector<int>{input_dim, head_dim});
        value_weights =
            std::make_shared<Tensor>(std::vector<int>{input_dim, head_dim});
        output_weights =
            std::make_shared<Tensor>(std::vector<int>{head_dim, head_dim});

        // 随机初始化权重
        std::srand(static_cast<unsigned int>(std::time(0)));
        for (size_t i = 0; i < query_weights->Size(); ++i) {
            query_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
            key_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
            value_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
            output_weights->data[i] =
                static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    SelfAttention(int input_dim, int head_dim,
                  const std::shared_ptr<Tensor> query_weights,
                  const std::shared_ptr<Tensor> key_weights,
                  const std::shared_ptr<Tensor> value_weights,
                  const std::shared_ptr<Tensor> output_weights)
        : input_dim(input_dim), head_dim(head_dim),
          query_weights(query_weights), key_weights(key_weights),
          value_weights(value_weights), output_weights(output_weights) {}

    ~SelfAttention() {}

  protected:
    std::shared_ptr<Tensor> query_weights;  // 查询权重
    std::shared_ptr<Tensor> key_weights;    // 键权重
    std::shared_ptr<Tensor> value_weights;  // 值权重
    std::shared_ptr<Tensor> output_weights; // 输出权重
    // int num_heads;                           // 注意力头数
    int head_dim;  // 每个头的维度
    int input_dim; // 输入维度

    std::shared_ptr<Tensor> grad_query_weights;
    std::shared_ptr<Tensor> grad_key_weights;
    std::shared_ptr<Tensor> grad_value_weights;
    std::shared_ptr<Tensor> grad_output_weights;
};
