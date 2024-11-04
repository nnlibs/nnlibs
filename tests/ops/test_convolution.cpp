// clang-format off
#include <iostream>

#include "ops/convolution/conv_2d_cpu.h"

int main() {
    // input data
    std::shared_ptr<Tensor> t_ptr =
        std::make_shared<Tensor>(std::vector<int>{1, 3, 7, 7});
    t_ptr->data = {
     0, 0, 0, 0, 0, 0,0, 
     0, 0, 1, 1, 0, 2,0,
     0, 2, 2, 2, 2, 1,0,
     0, 1, 0, 0, 2, 0,0,
     0, 0, 1, 1, 0, 0,0,
     0, 1, 2, 0, 0, 2,0,
     0, 0, 0, 0, 0, 0,0, 

     0, 0, 0, 0, 0, 0,0, 
     0, 1, 0, 2, 2, 0,0,
     0, 0, 0, 0, 2, 0,0,
     0, 1, 2, 1, 2, 1,0,
     0, 1, 0, 0, 0, 0,0,
     0, 1, 2, 1, 1, 1,0,
     0, 0, 0, 0, 0, 0,0, 

     0, 0, 0, 0, 0, 0,0, 
     0, 2, 1, 2, 0, 0,0,
     0, 1, 0, 0, 1, 0,0,
     0, 0, 2, 1, 0, 1,0,
     0, 0, 1, 2, 2, 2,0,
     0, 2, 1, 0, 0, 1,0,
     0, 0, 0, 0, 0, 0,0, 
    };
    std::shared_ptr<Tensor> grad_output =
        std::make_shared<Tensor>(std::vector<int>{1, 2, 3, 3});
    grad_output->data = {0, 1, 1, 2, 2, 2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 2, 1};

    std::shared_ptr<Tensor> weights =
        std::make_shared<Tensor>(std::vector<int>{2, 3, 3, 3});

    weights->data = {-1, 1,  0,  0,  1, 0, 0,  1, 1, -1, -1, 0,  0,  0,
                     0,  0,  -1, 0,  0, 0, -1, 0, 1, 0,  1,  -1, -1, 1,
                     1,  -1, -1, -1, 1, 0, -1, 1, 0, 1,  0,  -1, 0,  -1,
                     -1, 1,  0,  -1, 0, 0, -1, 0, 1, -1, 0,  0};
    std::shared_ptr<Tensor> bias =
        std::make_shared<Tensor>(std::vector<int>{2});
    bias->data = {1, 0};

    int kernel_size = 3;
    int in_channel = 3;
    int out_channel = 2;
    Conv2d *conv = new Conv2dCPU(in_channel, out_channel, kernel_size,
                                 kernel_size, 2, 0, weights, bias);

    auto o = conv->Forward(t_ptr);
    std::cout << o << std::endl;

    // auto param = conv->Parameters();
    auto b = conv->Backward(grad_output, 0.001);
    // std::cout << b << std::endl;
    auto param = conv->Parameters();

    // -1.002,0.994,-0.006,
    // -0.009,0.985,-0.013,
    // -0.006,0.994,0.991,

    // -1.002,-1.006,-0.008,
    // -0.001,-0.003,-0.01,
    // -0.006,-1.009,-0.012,

    // -0.006,-0.011,-1.006,
    // -0.005,0.995,-0.006,
    // 0.995,-1.006,-1.008,


    // 0.994,0.992,-1.008,
    // -1.004,-1.003,0.997,
    // -0.005,-1.009,0.991,

    // 0,1,-0.002,
    // -1.004,-0.011,-1.01,
    // -1.002,0.999,-0.004,

    // -1.002,-0.001,-0.001,
    // -1.004,-0.011,0.995,
    // -1.001,-0.005,-0.009

    return 0;
}
