#include <iostream>

#include "ops/attention/multi_head_attention_cpu.h"
#include "ops/attention/self_attention_cpu.h"

int main() {
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
  std::cout << "conv output shape: ";
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

  // test MHA
  std::cout << "=============test MHA============\n";

  return 0;
}