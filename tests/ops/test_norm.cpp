#include <ctime>
#include <iostream>

#include "ops/normalization/layer_norm_cpu.h"

int main() {
  std::vector<int> norm_shape{8, 3, 4, 4};
  std::shared_ptr<Tensor> t_ptr = std::make_shared<Tensor>(norm_shape);
  auto &t = *t_ptr;
  std::srand(static_cast<unsigned int>(std::time(0)));
  for (int i = 0; i < t_ptr->Size(); ++i) {
    t_ptr->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }

  LayerNorm *r = new LayerNormCPU(norm_shape);
  auto o = r->Forward(t_ptr);

  for (auto s : o->shape) {
    std::cout << s << " ";
  }
  std::cout << std::endl;

  int layer_size = o->Size() / o->shape[0];
  for (int i = 0; i < o->shape[0]; ++i) {
    for (int j = 0; j < layer_size; ++j) {
      std::cout << o->data[i * layer_size + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}