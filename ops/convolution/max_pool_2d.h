#pragma once

#include <iostream>
#include <random>

#include "operator.h"

class MaxPool2d : public Operator {
  public:
    MaxPool2d(int kernel_size, int stride)
        : kernel_size(kernel_size), stride(stride) {}

    std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> input) {
        std::cout << "MaxPool2d forward not implement" << std::endl;
    }

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                     float learning_rate) {
        std::cout << "MaxPool2d backward not implement" << std::endl;
    }

    ~MaxPool2d() {}

  protected:
    int kernel_size;
    int stride;
    int in_channels;
    int in_height;
    int in_width;

    int out_height;
    int out_width;

    std::shared_ptr<Tensor> max_h_index;
    std::shared_ptr<Tensor> max_w_index;
};
