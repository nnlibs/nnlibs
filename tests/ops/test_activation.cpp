#include <iostream>

#include "ops/activation/relu_cpu.h"
#include "ops/activation/sigmoid_cpu.h"

int main() {
  std::shared_ptr<Tensor> t_ptr =
      std::make_shared<Tensor>(std::vector<int>{2, 3});
  auto &t = *t_ptr;
  t({0, 0}) = 1;
  t({0, 1}) = 2;
  t({0, 2}) = -3;
  t({1, 0}) = 4;
  t({1, 1}) = -5;
  t({1, 2}) = 6;

  Relu *r = new ReluCPU();
  auto o = r->Forward(t_ptr);

  for (int i = 0; i < o->Size(); ++i) {
    std::cout << o->data[i] << " ";
  }
  std::cout << std::endl;

  Sigmoid *s = new SigmoidCPU();
  auto o2 = s->Forward(t_ptr);
  for (int i = 0; i < o2->Size(); ++i) {
    std::cout << o2->data[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}