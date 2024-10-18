#include <iostream>

#include "common/tensor.h"

int main() {
  // 创建一个形状为 2x3 的张量
  Tensor t({2, 3});
  t({0, 0}) = 1;
  t({0, 1}) = 2;
  t({0, 2}) = 3;
  t({1, 0}) = 4;
  t({1, 1}) = 5;
  t({1, 2}) = 6;

  std::cout << "Original Tensor:" << std::endl;
  auto shape = t.shape;
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      std::cout << t({i, j}) << " ";
    }
    std::cout << std::endl;
  }

  t({0, 1}) = 10;
  t({1, 2}) = 20;

  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      std::cout << t({i, j}) << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}