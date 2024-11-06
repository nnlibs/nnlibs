#include "conv_2d_cpu.h"
#include "common/tensor.h"
#include <memory>

Conv2dCPU::Conv2dCPU(int in_channel, int out_channel, int kernel_h,
                     int kernel_w, int stride, int padding,
                     const std::shared_ptr<Tensor> weights,
                     const std::shared_ptr<Tensor> bias)
    : Conv2d(in_channel, out_channel, kernel_h, kernel_w, stride, padding,
             weights, bias) {}

Conv2dCPU::Conv2dCPU(int in_channel, int out_channel, int kernel_h,
                     int kernel_w, int stride, int padding)
    : Conv2d(in_channel, out_channel, kernel_h, kernel_w, stride, padding) {}

Conv2dCPU::Conv2dCPU(int in_channel, int out_channel, int kernel_size)
    : Conv2d(in_channel, out_channel, kernel_size) {}

std::shared_ptr<Tensor>
Conv2dCPU::Forward(const std::shared_ptr<Tensor> input) {
    in_height = input->shape[2];
    in_width = input->shape[3];

    input_bakup = input;
    out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(
        std::vector<int>{1, out_channel, out_height, out_width});
    auto &output = *result;

    auto &w_data = *weights;
    auto &in_data = *input;
    for (int oc = 0; oc < out_channel; oc++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                float val = bias->data[oc];
                for (int ic = 0; ic < in_channel; ic++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            val += w_data({oc, ic, kh, kw}) *
                                   in_data({0, ic, h * stride + kh - padding,
                                            w * stride + kw - padding});
                        }
                    }
                }
                output({0, oc, h, w}) = val; // 0 means batch is 1
            }
        }
    }

    return result;
}

std::shared_ptr<Tensor>
Conv2dCPU::Backward(const std::shared_ptr<Tensor> grad_output,
                    float learning_rate, float momentum) {
    // 计算输入的梯度
    std::shared_ptr<Tensor> grad_input = std::make_shared<Tensor>(
        std::vector<int>{1, in_channel, in_height, in_width});

    // 计算权重和偏置的梯度
    std::shared_ptr<Tensor> grad_weights = std::make_shared<Tensor>(
        std::vector<int>{out_channel, in_channel, kernel_h, kernel_w});
    std::shared_ptr<Tensor> grad_biases =
        std::make_shared<Tensor>(std::vector<int>{out_channel});

    auto &grad_weights_mut = *grad_weights;
    auto &grad_input_mut = *grad_input;
    auto &grad_output_mut = *grad_output;
    auto &weights_mut = *weights;
    auto &input_bakup_mut = *input_bakup;

    // calculate the gradient of input/weight/bias
    for (int o = 0; o < out_channel; ++o) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                auto grad = grad_output_mut({0, o, h, w});

                // 计算输入的梯度
                for (int c = 0; c < in_channel; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            grad_input_mut({0, c, h * stride + kh - padding,
                                            w * stride + kw - padding}) +=
                                grad * weights_mut({o, c, kh, kw});
                            grad_weights_mut({o, c, kh, kw}) +=
                                grad * input_bakup_mut(
                                           {0, c, h * stride + kh - padding,
                                            w * stride + kw - padding});
                        }
                    }
                }
                grad_biases->data[o] += grad;
            }
        }
    }

    // update weight/bias
    for (int i = 0; i < grad_weights->Size(); i++) {
        weights_momentum->data[i] = momentum * weights_momentum->data[i] +
                                    (1 - momentum) * grad_weights->data[i];
        weights->data[i] -= learning_rate * weights_momentum->data[i];
    }
    for (int i = 0; i < grad_biases->Size(); i++) {
        bias_momentum->data[i] = momentum * bias_momentum->data[i] +
                                 (1 - momentum) * grad_biases->data[i];
        bias->data[i] -= learning_rate * bias_momentum->data[i];
    }

    return grad_input;
}
std::shared_ptr<Tensor> Conv2dCPU::Parameters() {
    std::cout << "Conv2d Parameters [" << std::endl;
    std::cout << "-- Weight is: " << weights << std::endl;
    std::cout << "-- Bias is: " << bias << std::endl;
    std::cout << "]" << std::endl;
    return nullptr;
}

void Conv2dCPU::ZeroGrad() {}

Conv2dCPU::~Conv2dCPU() {}
