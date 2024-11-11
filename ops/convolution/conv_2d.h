#pragma once

#include <ctime>
#include <iostream>
#include <memory>

#include "common/tensor.h"
#include "functional/functional.h"
#include "operator.h"

class Conv2d : public Operator {
  public:
    Conv2d(int in_channel, int out_channel, int kernel_h, int kernel_w,
           int stride, int padding, const std::shared_ptr<Tensor> weights,
           const std::shared_ptr<Tensor> bias)
        : in_channel(in_channel), out_channel(out_channel), kernel_h(kernel_h),
          kernel_w(kernel_w), stride(stride), padding(padding),
          weights(weights), bias(bias) {
        if (this->weights == nullptr) {
            this->weights = std::make_shared<Tensor>(
                std::vector<int>{out_channel, in_channel, kernel_h, kernel_w});
            // weight use Kaiming initialization
            int fan_in = in_channel * kernel_h * kernel_w;
            F::kaiming_normal(this->weights, fan_in);
        }
        if (this->bias == nullptr) {
            this->bias =
                std::make_shared<Tensor>(std::vector<int>{out_channel});
            for (size_t i = 0; i < this->bias->Size(); ++i) {
                this->bias->data[i] = 0.0f; // 偏置初始化为0
            }
        }
        weights_momentum = std::make_shared<Tensor>(
            std::vector<int>{out_channel, in_channel, kernel_h, kernel_w});
        bias_momentum = std::make_shared<Tensor>(std::vector<int>{out_channel});
        kernel_area = kernel_h * kernel_w;
    }

    Conv2d(int in_channel, int out_channel, int kernel_h, int kernel_w,
           int stride, int padding)
        : Conv2d(in_channel, out_channel, kernel_h, kernel_w, stride, padding,
                 nullptr, nullptr) {}

    Conv2d(int in_channel, int out_channel, int kernel_size)
        : Conv2d(in_channel, out_channel, kernel_size, kernel_size, 1, 0) {}

    std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> input) {
        std::cout << "Conv2d forward not implement" << std::endl;
        return nullptr;
    }

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                     float learning_rate, float momentum) {
        std::cout << "Conv2d backward not implement" << std::endl;
        return nullptr;
    }

    std::shared_ptr<Tensor> Parameters() {
        std::cout << "Conv2d backward not implement" << std::endl;
        return nullptr;
    }

    ~Conv2d() {}

  protected:
    // [out_channel, in_channel, kernel_h, kernel_w]
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> weights_momentum;
    std::shared_ptr<Tensor> bias_momentum;
    int stride;
    int padding;
    int in_channel;
    int out_channel;
    int kernel_h;
    int kernel_w;
    int kernel_area;

    int in_height;
    int in_width;
    int out_height;
    int out_width;
    std::shared_ptr<Tensor> input_bakup;
};
