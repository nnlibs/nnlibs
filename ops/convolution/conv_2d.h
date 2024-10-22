#pragma once

#include <ctime>
#include <iostream>

#include "operator.h"

class Conv2d : public Operator {
 public:
  Conv2d(int in_channel, int out_channel, int kernel_h, int kernel_w,
         int stride, int padding, const std::shared_ptr<Tensor> weights,
         const std::shared_ptr<Tensor> bias)
      : in_channel(in_channel),
        out_channel(out_channel),
        kernel_h(kernel_h),
        kernel_w(kernel_w),
        stride(stride),
        padding(padding),
        weights(weights),
        bias(bias) {
    grad_w = std::make_shared<Tensor>(weights->shape);
    grad_b = std::make_shared<Tensor>(bias->shape);
  }

  Conv2d(int in_channel, int out_channel, int kernel_h, int kernel_w,
         int stride, int padding)
      : in_channel(in_channel),
        out_channel(out_channel),
        kernel_h(kernel_h),
        kernel_w(kernel_w),
        stride(stride),
        padding(padding) {
    weights = std::make_shared<Tensor>(
        std::vector<int>{out_channel, in_channel, kernel_h, kernel_w});
    bias = std::make_shared<Tensor>(std::vector<int>{out_channel});
    // 随机初始化权重和偏置
    std::srand(static_cast<unsigned int>(std::time(0)));
    for (size_t i = 0; i < weights->Size(); ++i) {
      weights->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < bias->Size(); ++i) {
      bias->data[i] = 0.0f;  // 偏置初始化为0
    }
  }

  std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> input) {
    std::cout << "Conv2d forward not implement" << std::endl;
  }

  std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                   float learning_rate) {
    std::cout << "Conv2d backward not implement" << std::endl;
  }

  ~Conv2d() {}

 protected:
  // [out_channel, in_channel, kernel_h, kernel_w]
  std::shared_ptr<Tensor> weights;
  std::shared_ptr<Tensor> bias;
  int stride;
  int padding;
  int in_channel;
  int out_channel;
  int kernel_h;
  int kernel_w;

  std::shared_ptr<Tensor> grad_w;
  std::shared_ptr<Tensor> grad_b;
};