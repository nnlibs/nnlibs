#include "max_pool_2d_cpu.h"
#include <cmath>

MaxPool2dCpu::MaxPool2dCpu(int kernel_size, int stride)
    : MaxPool2d(kernel_size, stride) {}

std::shared_ptr<Tensor>
MaxPool2dCpu::Forward(const std::shared_ptr<Tensor> input) {
    if (stride == -1) {
        stride = kernel_size;
    }
    assert(input->shape[0] == 1); // batch size only support one an now
    in_channels = input->shape[1];
    in_height = input->shape[2];
    in_width = input->shape[3];

    out_height = (in_height - kernel_size) / stride + 1;
    out_width = (in_width - kernel_size) / stride + 1;

    std::shared_ptr<Tensor> output = std::make_shared<Tensor>(
        std::vector<int>{input->shape[0], in_channels, out_height, out_width});
    max_h_index = std::make_shared<Tensor>(
        std::vector<int>{input->shape[0], in_channels, out_height, out_width});
    max_w_index = std::make_shared<Tensor>(
        std::vector<int>{input->shape[0], in_channels, out_height, out_width});

    auto &input_mut = *input;
    auto &output_mut = *output;
    auto &max_h_index_mut = *max_h_index;
    auto &max_w_index_mut = *max_w_index;
    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                float max_val = -100000000;
                int max_h_index_s = -1;
                int max_w_index_s = -1;
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h_in = h * stride + kh;
                        int w_in = w * stride + kw;
                        if (input_mut({0, c, h_in, w_in}) > max_val) {
                            max_val = input_mut({0, c, h_in, w_in});
                            max_h_index_s = h_in;
                            max_w_index_s = w_in;
                        }
                    }
                }
                output_mut({0, c, h, w}) = max_val;
                max_h_index_mut({0, c, h, w}) = max_h_index_s;
                max_w_index_mut({0, c, h, w}) = max_w_index_s;
            }
        }
    }

    return output;
}

std::shared_ptr<Tensor>
MaxPool2dCpu::Backward(const std::shared_ptr<Tensor> grad_output,
                       float learning_rate) {
    std::shared_ptr<Tensor> grad_input = std::make_shared<Tensor>(
        std::vector<int>{1, in_channels, in_height, in_width});

    // 执行反向传播
    auto &max_h_index_mut = *max_h_index;
    auto &max_w_index_mut = *max_w_index;
    auto &grad_input_mut = *grad_input;
    auto &grad_output_mut = *grad_output;
    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                // 获取最大值的位置
                int max_h = static_cast<int>(max_h_index_mut({0, c, h, w}));
                int max_w = static_cast<int>(max_w_index_mut({0, c, h, w}));
                grad_input_mut({0, c, max_h, max_w}) +=
                    grad_output_mut({0, c, h, w});
            }
        }
    }

    return grad_input;
}
std::shared_ptr<Tensor> MaxPool2dCpu::Parameters() { return nullptr; }

void MaxPool2dCpu::ZeroGrad() {
    max_h_index.reset();
    max_w_index.reset();
}

MaxPool2dCpu::~MaxPool2dCpu() {}
