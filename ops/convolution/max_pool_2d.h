#pragma once

#include <iostream>

#include "operator.h"

class MaxPool2d : public Operator {
  public:
    MaxPool2d(int kernel_size, int stride)
        : kernel_size(kernel_size), stride(stride) {
        kernel_area = kernel_size * kernel_size;
    }

    std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> input) {
        std::cout << "MaxPool2d forward not implement" << std::endl;
    }

    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> output,
                                     float learning_rate, float momentum) {
        std::cout << "MaxPool2d backward not implement" << std::endl;
    }

    ~MaxPool2d() {}

  protected:
    int kernel_size;
    int kernel_area;
    int stride;
    int in_channels;
    int in_height;
    int in_width;

    int out_height;
    int out_width;

    std::shared_ptr<Tensor> max_index;
};
