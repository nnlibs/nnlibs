#include "conv_2d_cpu.h"
#include "common/tensor.h"
#include "functional/blas.h"
#include <memory>
#include <x86intrin.h>

static std::shared_ptr<Tensor>
Im2Col_Optimized(const std::shared_ptr<Tensor> input, int in_channel,
                 int input_height, int input_width, int kernel_h, int kernel_w,
                 int stride, int padding, int output_height, int output_width) {
    // pre padding to avoid runtime if check
    int padded_height = input_height + 2 * padding;
    int padded_width = input_width + 2 * padding;
    auto padded_input = std::make_shared<Tensor>(
        std::vector<int>{1, in_channel, padded_height, padded_width});

// OpenMp acceleration
#pragma omp parallel for collapse(3) num_threads(4)
    for (int c = 0; c < in_channel; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                padded_input
                    ->data[(c * padded_height + (h + padding)) * padded_width +
                           (w + padding)] =
                    input->data[(c * input_height + h) * input_width + w];
            }
        }
    }

    std::shared_ptr<Tensor> col_buffer = // [K,N]
        std::make_shared<Tensor>(std::vector<int>{
            in_channel * kernel_h * kernel_w, output_height * output_width});
    int kernel_area = kernel_h * kernel_w;
// OpenMp acceleration
#pragma omp parallel for collapse(3) num_threads(4)
    for (int c = 0; c < in_channel; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int row_offset = c * kernel_area + kh * kernel_w + kw;
                for (int h = 0; h < output_height; ++h) {
                    // SIMD acceleration
                    int w = 0;
                    for (; w <= output_width - 8; w += 8) {
                        int h_in = h * stride + kh;
                        int w_in = w * stride + kw;
                        // TODO: support any stride. now  support stride=1 only
                        __m256 target = _mm256_loadu_ps(
                            &padded_input->data[(c * padded_height + h_in) *
                                                    padded_width +
                                                w_in]);
                        _mm256_storeu_ps(
                            &col_buffer->data[(row_offset * output_height + h) *
                                                  output_width +
                                              w],
                            target);
                    }
                    for (; w < output_width; ++w) {
                        int h_in = h * stride + kh;
                        int w_in = w * stride + kw;
                        col_buffer->data[(row_offset * output_height + h) *
                                             output_width +
                                         w] =
                            padded_input->data[(c * padded_height + h_in) *
                                                   padded_width +
                                               w_in];
                    }
                }
            }
        }
    }
    return col_buffer;
}

std::shared_ptr<Tensor> Col2Im_Optimized(const std::shared_ptr<Tensor> col,
                                         int in_channel, int input_height,
                                         int input_width, int kernel_h,
                                         int kernel_w, int stride, int padding,
                                         int output_height, int output_width) {
    auto im_data = std::make_shared<Tensor>(
        std::vector<int>{1, in_channel, input_height, input_width});

#pragma omp parallel for collapse(3) num_threads(4)
    for (int c = 0; c < in_channel; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int row_offset = (c * kernel_h + kh) * kernel_w + kw;
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        int h = oh * stride - padding + kh;
                        int w = ow * stride - padding + kw;
                        int col_index =
                            (row_offset * output_height + oh) * output_width +
                            ow;

                        int im_index = (c * input_height + h) * input_width + w;

#pragma omp atomic
                        im_data->data[im_index] += col->data[col_index];
                    }
                }
            }
        }
    }
    return im_data;
}

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

    out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

    col_buffer =
        Im2Col_Optimized(input, in_channel, in_height, in_width, kernel_h,
                         kernel_w, stride, padding, out_height, out_width);

    weights->View(
        std::vector<int>{out_channel, in_channel * kernel_h * kernel_w});
    auto gemm_output = F::TensorMul(weights, col_buffer);
    gemm_output->View(std::vector<int>{1, out_channel, out_height, out_width});

    // add bias in each channel
    int out_area = out_height * out_width;
    for (int oc = 0; oc < out_channel; ++oc) {
        __m256 bias_v = _mm256_set1_ps(bias->data[oc]);
        int hw = 0;
        for (; hw <= out_area - 8; hw += 8) {
            __m256 output_v =
                _mm256_loadu_ps(&gemm_output->data[oc * out_area + hw]);
            _mm256_storeu_ps(&gemm_output->data[oc * out_area + hw],
                             _mm256_add_ps(output_v, bias_v));
        }

        for (; hw < out_area; ++hw) {
            gemm_output->data[oc * out_area + hw] += bias->data[oc];
        }
    }

    return gemm_output;
}

std::shared_ptr<Tensor>
Conv2dCPU::Backward(const std::shared_ptr<Tensor> grad_output,
                    float learning_rate, float momentum) {
    // grad_output: {1, out_channel, out_height, out_width}
    // col_buffer: {in_channel * kernel_h * kernel_w, out_height * out_width}
    // grad_weight=grad_output x col_buffer_T
    grad_output->View({out_channel, out_height * out_width});
    auto grad_weights = F::TensorMul(grad_output, col_buffer, false, true);

    // weights: {out_channel, in_channel * kernel_h * kernel_w}
    // grad_output: {1, out_channel, out_height, out_width}
    // grad_input=weights_T x grad_output
    weights->View({out_channel, in_channel * kernel_h * kernel_w});
    auto grad_col_input = F::TensorMul(weights, grad_output, true, false);

    auto grad_input = Col2Im_Optimized(grad_col_input, in_channel, in_height,
                                       in_width, kernel_h, kernel_w, stride,
                                       padding, out_height, out_width);

    auto one_tensor = std::make_shared<Tensor>(
        std::vector<int>{1, out_height * out_width}, 1.0f);
    auto grad_biases = F::TensorMul(one_tensor, grad_output, false, true);

    grad_output->View({1, out_channel, out_height, out_width});

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
