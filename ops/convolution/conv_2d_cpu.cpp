#include "conv_2d_cpu.h"

Conv2dCPU::Conv2dCPU(int in_channel, int out_channel, int kernel_h,
                     int kernel_w, int stride, int padding,
                     const std::shared_ptr<Tensor> weights,
                     const std::shared_ptr<Tensor> bias)
    : Conv2d(in_channel, out_channel, kernel_h, kernel_w, stride, padding,
             weights, bias) {}

Conv2dCPU::Conv2dCPU(int in_channel, int out_channel, int kernel_h,
                     int kernel_w, int stride, int padding)
    : Conv2d(in_channel, out_channel, kernel_h, kernel_w, stride, padding) {}

std::shared_ptr<Tensor> Conv2dCPU::Forward(
    const std::shared_ptr<Tensor> input) {
  int in_height = input->shape[2];
  int in_width = input->shape[3];

  // (in_size - kernel_size + 2*padding) / stride + 1
  int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
  int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

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
                     in_data({0, ic, h * stride + kh, w * stride + kw});
            }
          }
        }
        output({0, oc, h, w}) = val;  // 0 means batch is 1
      }
    }
  }

  return result;
}

std::shared_ptr<Tensor> Conv2dCPU::Backward(
    const std::shared_ptr<Tensor> output) {
  return nullptr;
}

Conv2dCPU::~Conv2dCPU() {}