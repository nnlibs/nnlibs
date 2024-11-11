#include "max_pool_2d_cpu.h"
#include <cmath>
#include <omp.h>

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

    max_index = std::make_shared<Tensor>(
        std::vector<int>{input->shape[0], in_channels, out_height, out_width});
    auto &input_mut = *input;
    auto &output_mut = *output;
    auto &max_index_mut = *max_index;
#pragma omp parallel for collapse(3) num_threads(4)
    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                float max_val = -std::numeric_limits<float>::infinity();
                int max_index_s = -1;
                int out_index = 0;
                output_mut.GetIndexByIndices({0, c, h, w}, out_index);
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int maybe_max_index = 0;
                        input_mut.GetIndexByIndices(
                            {0, c, h * stride + kh, w * stride + kw},
                            maybe_max_index);
                        if (input_mut[maybe_max_index] > max_val) {
                            max_val = input_mut[maybe_max_index];
                            max_index_s = maybe_max_index;
                        }
                    }
                }
                output_mut[out_index] = max_val;
                max_index_mut[out_index] = max_index_s;
            }
        }
    }

    return output;
}

std::shared_ptr<Tensor>
MaxPool2dCpu::Backward(const std::shared_ptr<Tensor> grad_output,
                       float learning_rate, float momentum) {
    std::shared_ptr<Tensor> grad_input = std::make_shared<Tensor>(
        std::vector<int>{1, in_channels, in_height, in_width});

    auto &max_index_mut = *max_index;
    auto &grad_input_mut = *grad_input;
    auto &grad_output_mut = *grad_output;

    auto size = in_channels * out_height * out_width;
    for (int i = 0; i < size; ++i) {
        grad_input_mut[max_index_mut[i]] += grad_output_mut[i];
    }
    return grad_input;
}

std::shared_ptr<Tensor> MaxPool2dCpu::Parameters() { return nullptr; }

void MaxPool2dCpu::ZeroGrad() { max_index.reset(); }

MaxPool2dCpu::~MaxPool2dCpu() {}
