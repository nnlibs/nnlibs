#include <iostream>

#include "ops/attention/multi_head_attention_cpu.h"
#include "ops/attention/self_attention_cpu.h"

int main() {
  {
    int dim = 32;
    // input data
    std::shared_ptr<Tensor> t_ptr =
        std::make_shared<Tensor>(std::vector<int>{5, dim});
    std::srand(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < t_ptr->Size(); ++i) {
      t_ptr->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // test self-attention
    std::cout << "=============test self-attention============\n";

    std::shared_ptr<Tensor> q_weights =
        std::make_shared<Tensor>(std::vector<int>{dim, dim});
    std::shared_ptr<Tensor> k_weights =
        std::make_shared<Tensor>(std::vector<int>{dim, dim});
    std::shared_ptr<Tensor> v_weights =
        std::make_shared<Tensor>(std::vector<int>{dim, dim});
    std::shared_ptr<Tensor> o_weights =
        std::make_shared<Tensor>(std::vector<int>{dim, dim});

    for (int i = 0; i < q_weights->Size(); ++i) {
      q_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
      k_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
      v_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
      o_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    SelfAttention *atte = new SelfAttentionCPU(dim, dim, q_weights, k_weights,
                                               v_weights, o_weights);
    auto o = atte->Forward(t_ptr);
    std::cout << "self_atte output shape: ";
    for (auto s : o->shape) {
      std::cout << s << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < o->shape[0]; ++i) {
      for (int j = 0; j < o->shape[1]; ++j) {
        std::cout << o->data[i * o->shape[1] + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  // test MHA
  {
    std::cout << "=============test MHA============\n";
    int vec_dim = 128;
    int num_heads = 8;
    int head_dim = vec_dim / num_heads;
    // std::srand(static_cast<unsigned int>(std::time(0)));
    std::vector<std::shared_ptr<Tensor>> q_weights_list;
    std::vector<std::shared_ptr<Tensor>> k_weights_list;
    std::vector<std::shared_ptr<Tensor>> v_weights_list;
    std::vector<std::shared_ptr<Tensor>> o_weights_list;
    std::shared_ptr<Tensor> concat_out_weights =
        std::make_shared<Tensor>(std::vector<int>{vec_dim, vec_dim});
    for (int i = 0; i < concat_out_weights->Size(); ++i) {
      concat_out_weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    std::shared_ptr<Tensor> t_ptr =
        std::make_shared<Tensor>(std::vector<int>{5, vec_dim});
    for (int i = 0; i < t_ptr->Size(); ++i) {
      t_ptr->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (int i = 0; i < num_heads; ++i) {
      q_weights_list.emplace_back(
          std::make_shared<Tensor>(std::vector<int>{vec_dim, head_dim}));
      k_weights_list.emplace_back(
          std::make_shared<Tensor>(std::vector<int>{vec_dim, head_dim}));
      v_weights_list.emplace_back(
          std::make_shared<Tensor>(std::vector<int>{vec_dim, head_dim}));
      o_weights_list.emplace_back(
          std::make_shared<Tensor>(std::vector<int>{vec_dim, head_dim}));
      for (int j = 0; j < q_weights_list[i]->Size(); ++j) {
        q_weights_list[i]->data[j] = static_cast<float>(std::rand()) / RAND_MAX;
        k_weights_list[i]->data[j] = static_cast<float>(std::rand()) / RAND_MAX;
        v_weights_list[i]->data[j] = static_cast<float>(std::rand()) / RAND_MAX;
        o_weights_list[i]->data[j] = static_cast<float>(std::rand()) / RAND_MAX;
      }
    }
    MultiHeadAttention *mha = new MultiHeadAttentionCPU(
        vec_dim, num_heads, q_weights_list, k_weights_list, v_weights_list,
        o_weights_list, concat_out_weights);
    auto o = mha->Forward(t_ptr);
    std::cout << "mha output shape: ";
    for (auto s : o->shape) {
      std::cout << s << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < o->shape[0]; ++i) {
      for (int j = 0; j < o->shape[1]; ++j) {
        std::cout << o->data[i * o->shape[1] + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  return 0;
}