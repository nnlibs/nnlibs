#pragma once
#include "common/tensor.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <x86intrin.h>

namespace F {
static std::shared_ptr<Tensor> max_pool2d(const std::shared_ptr<Tensor> input,
                                          int kernel_size, int stride = -1) {
    if (stride == -1) {
        stride = kernel_size;
    }
    assert(input->shape[0] == 1); // batch size only support one an now
    int in_channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];

    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;

    std::shared_ptr<Tensor> output = std::make_shared<Tensor>(
        std::vector<int>{input->shape[0], in_channels, out_height, out_width});

    auto &input_mut = *input;
    auto &output_mut = *output;
    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                float max_val = -2147483647 - 1;
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h_in = h * stride + kh;
                        int w_in = w * stride + kw;
                        max_val =
                            std::max(max_val, input_mut({0, c, h_in, w_in}));
                    }
                }
                output_mut({0, c, h, w}) = max_val;
            }
        }
    }

    return output;
}

static std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor> input) {
    assert(input->shape[0] == 1); // batch size only support one an now
    int in_channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];

    std::shared_ptr<Tensor> output = std::make_shared<Tensor>(
        std::vector<int>{input->shape[0], in_channels, in_height, in_width});

    auto &input_mut = *input;
    auto &output_mut = *output;
    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < in_height; ++h) {
            for (int w = 0; w < in_width; ++w) {
                output_mut({0, c, h, w}) = std::max(
                    0.0f, input_mut({0, c, h, w})); // cut less 0 -> ReLU
            }
        }
    }

    return output;
}

static float MSELoss(const std::shared_ptr<Tensor> input,
                     const std::shared_ptr<Tensor> target) {
    assert(input->shape == target->shape);
    float loss = 0.0;
    for (int i = 0; i < input->Size(); ++i) {
        loss += (input->data[i] - target->data[i]) *
                (input->data[i] - target->data[i]);
    }
    return loss / input->Size();
}

static void Normalize(std::shared_ptr<Tensor> input,
                      const std::array<float, 3> &mean,
                      const std::array<float, 3> &std) {
    assert(input->shape.size() == 4);
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];

    auto &input_mut = *input;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                input_mut({0, c, h, w}) =
                    (input_mut({0, c, h, w}) - mean[c]) / std[c];
            }
        }
    }
}

static void kaiming_normal(std::shared_ptr<Tensor> weights, int fan_in) {
    // 计算标准差
    float std_dev = std::sqrt(2.0f / fan_in);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, std_dev);

    // 按标准正态分布初始化权重
    for (int i = 0; i < weights->Size(); ++i) {
        weights->data[i] = distribution(generator);
    }
}

static std::shared_ptr<Tensor> Im2Col(const std::shared_ptr<Tensor> input,
                                      int in_channel, int input_height,
                                      int input_width, int kernel_h,
                                      int kernel_w, int stride, int padding,
                                      int output_height, int output_width) {
    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<Tensor> col_buffer = // [K,N]
        std::make_shared<Tensor>(std::vector<int>{
            in_channel * kernel_h * kernel_w, output_height * output_width});
    int kernel_area = kernel_h * kernel_w;
    int out_area = output_height * output_width;
    int input_area = input_height * input_width;
    for (int c = 0; c < in_channel; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int h = 0; h < output_height; ++h) {
                    for (int w = 0; w < output_width; ++w) {
                        int h_in = h * stride - padding + kh;
                        int w_in = w * stride - padding + kw;
                        if (h_in >= 0 && h_in < input_height && w_in >= 0 &&
                            w_in < input_width) {
                            col_buffer
                                ->data[(c * kernel_area + kh * kernel_w + kw) *
                                           out_area +
                                       h * output_width + w] =
                                input->data[c * input_area +
                                            h_in * input_width + w_in];
                        } else {
                            col_buffer
                                ->data[(c * kernel_area + kh * kernel_w + kw) *
                                           out_area +
                                       h * output_width + w] = 0.0f;
                        }
                    }
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Im2Col time: " << duration.count() << "us" << std::endl;
    return col_buffer;
}

} // namespace F
