#include <iostream>

#include "ops/convolution/conv_2d_cpu.h"
#include "ops/convolution/linear_cpu.h"

int main() {
  // input data
  std::shared_ptr<Tensor> t_ptr =
      std::make_shared<Tensor>(std::vector<int>{1, 3, 16, 16});
  std::srand(static_cast<unsigned int>(std::time(0)));
  for (int i = 0; i < t_ptr->Size(); ++i) {
    t_ptr->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }

  // test conv 2d
  std::cout << "=============test conv 2d============\n";
  int kernel_h = 3;
  int kernel_w = 3;
  int stride = 1;
  int padding = 0;
  int in_channel = 3;
  int out_channel = 16;
  std::shared_ptr<Tensor> weights = std::make_shared<Tensor>(
      std::vector<int>{out_channel, in_channel, kernel_h, kernel_w});
  for (int i = 0; i < weights->Size(); ++i) {
    weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }
  std::shared_ptr<Tensor> bias =
      std::make_shared<Tensor>(std::vector<int>{out_channel});
  Conv2d *conv = new Conv2dCPU(in_channel, out_channel, kernel_h, kernel_w,
                               stride, padding, weights, bias);
  auto o = conv->Forward(t_ptr);
  std::cout << "conv output shape: ";
  for (auto s : o->shape) {
    std::cout << s << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < o->Size(); ++i) {
    std::cout << o->data[i] << " ";
  }
  std::cout << std::endl;

  // test linear
  std::cout << "=============test linear============\n";
  int in_features = 16;
  int out_features = 10;
  std::shared_ptr<Tensor> linear_w =
      std::make_shared<Tensor>(std::vector<int>{out_features, in_features});
  for (int i = 0; i < linear_w->Size(); ++i) {
    linear_w->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }
  std::shared_ptr<Tensor> linear_b =
      std::make_shared<Tensor>(std::vector<int>{out_features});
  for (int i = 0; i < linear_b->Size(); ++i) {
    linear_b->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }
  Linear *linear = new LinearCPU(in_features, out_features, linear_w, linear_b);
  auto o2 = linear->Forward(o);

  for (int i = 0; i < o2->Size(); ++i) {
    std::cout << o2->data[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}