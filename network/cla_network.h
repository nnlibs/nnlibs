#pragma once

#include "activation/relu_cpu.h"
#include "activation/sigmoid_cpu.h"
#include "convolution/conv_2d_cpu.h"
#include "convolution/max_pool_2d_cpu.h"
#include "functional/functional.h"
#include "network.h"

class ClassifyNetwork : public Network {
  private:
    Conv2dCPU *conv1;
    Conv2dCPU *conv2;
    Linear *ln1;
    Linear *ln2;
    Linear *ln3;
    ReluCPU *relu1;
    ReluCPU *relu2;
    ReluCPU *relu3;
    ReluCPU *relu4;
    MaxPool2dCpu *max_pool1;
    MaxPool2dCpu *max_pool2;
    std::vector<int> shape;

  public:
    ClassifyNetwork(bool use_gpu = false) : Network(use_gpu) {
        if (use_gpu) {
        } else {
            conv1 = new Conv2dCPU(3, 6, 5);
            conv2 = new Conv2dCPU(6, 16, 5);
            ln1 = new LinearCPU(16 * 5 * 5, 120);
            ln2 = new LinearCPU(120, 84);
            ln3 = new LinearCPU(84, 10);
            relu1 = new ReluCPU();
            relu2 = new ReluCPU();
            relu3 = new ReluCPU();
            relu4 = new ReluCPU();
            max_pool1 = new MaxPool2dCpu(2);
            max_pool2 = new MaxPool2dCpu(2);
        }
    }

    std::shared_ptr<Tensor>
    // (1,3,32,32) -> (1,10)
    Forward(const std::shared_ptr<Tensor> input) override final {
        auto conv1_out = conv1->Forward(input);
        auto relu1_out = relu1->Forward(conv1_out);
        // std::cout << "relu1_out: " << relu1_out << std::endl;
        auto max_pool1_out = max_pool1->Forward(relu1_out);
        // std::cout << "max_pool1_out: " << max_pool1_out << std::endl;

        auto conv2_out = conv2->Forward(max_pool1_out);
        auto relu2_out = relu2->Forward(conv2_out);
        auto max_pool2_out = max_pool2->Forward(relu2_out);

        // reshape to (1, 16*5*5)
        shape = max_pool2_out->shape;
        max_pool2_out->shape = std::vector<int>{max_pool2_out->Size()};

        auto liner1_out = ln1->Forward(max_pool2_out);
        auto relu3_out = relu3->Forward(liner1_out);
        auto liner2_out = ln2->Forward(relu3_out);
        auto relu4_out = relu4->Forward(liner2_out);

        auto liner3_out = ln3->Forward(relu4_out);
        return liner3_out;
    }

    void Backward(const std::shared_ptr<Tensor> loss, float lr) override final {
        auto liner3_grad = ln3->Backward(loss, lr);
        // std::cout << "liner3_grad: " << liner3_grad << std::endl;
        auto relu4_grad = relu4->Backward(liner3_grad, lr);
        // std::cout << "relu4_grad: " << relu4_grad << std::endl;
        auto liner2_grad = ln2->Backward(relu4_grad, lr);
        // std::cout << "liner2_grad: " << liner2_grad << std::endl;
        auto relu3_grad = relu3->Backward(liner2_grad, lr);
        // std::cout << "relu3_grad: " << relu3_grad << std::endl;
        auto liner1_grad = ln1->Backward(relu3_grad, lr);
        // std::cout << "liner1_grad: " << liner1_grad << std::endl;

        // reshape to (1, 16, 5, 5)
        liner1_grad->shape = shape;

        auto max_pool2_grad = max_pool2->Backward(liner1_grad, lr);
        // std::cout << "max_pool2_grad: " << max_pool2_grad << std::endl;
        auto relu2_grad = relu2->Backward(max_pool2_grad, lr);
        // std::cout << "relu2_grad: " << relu2_grad << std::endl;
        auto conv2_grad = conv2->Backward(relu2_grad, lr);
        // std::cout << "conv2_grad: " << conv2_grad << std::endl;

        auto max_pool1_grad = max_pool1->Backward(conv2_grad, lr);
        // std::cout << "max_pool1_grad: " << max_pool1_grad << std::endl;
        auto relu1_grad = relu1->Backward(max_pool1_grad, lr);
        // std::cout << "relu1_grad: " << relu1_grad << std::endl;
        auto conv1_grad = conv1->Backward(relu1_grad, lr);
        // std::cout << "conv1_grad: " << conv1_grad << std::endl;
    }

    void ZeroGrad() {
        conv1->ZeroGrad();
        conv2->ZeroGrad();
        ln1->ZeroGrad();
        ln2->ZeroGrad();
        ln3->ZeroGrad();
        relu1->ZeroGrad();
        relu2->ZeroGrad();
        relu3->ZeroGrad();
        relu4->ZeroGrad();
        max_pool1->ZeroGrad();
        max_pool2->ZeroGrad();
    }

    ~ClassifyNetwork() {
        delete conv1;
        delete conv2;
        delete ln1;
        delete ln2;
        delete ln3;
        delete relu1;
        delete relu2;
        delete relu3;
        delete relu4;
        delete max_pool1;
        delete max_pool2;
    }
};
