#pragma once

#include "network.h"

class SimpleNetwork : public Network {
  private:
    Linear *fc1;
    Relu *relu;
    Linear *fc2;
    Sigmoid *sigmoid;

  public:
    SimpleNetwork(int input_size, int hidden_size, int output_size,
                  bool use_gpu = false)
        : Network(use_gpu) {
        std::shared_ptr<Tensor> fc1_w =
            std::make_shared<Tensor>(std::vector<int>{input_size, hidden_size});
        std::shared_ptr<Tensor> fc1_b =
            std::make_shared<Tensor>(std::vector<int>{hidden_size});
        auto fc2_w = std::make_shared<Tensor>(
            std::vector<int>{hidden_size, output_size});
        auto fc2_b = std::make_shared<Tensor>(std::vector<int>{output_size});

        std::srand(static_cast<unsigned int>(std::time(0)));
        for (int i = 0; i < fc1_w->Size(); ++i) {
            fc1_w->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }
        for (int i = 0; i < fc1_b->Size(); ++i) {
            fc1_b->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }
        for (int i = 0; i < fc2_w->Size(); ++i) {
            fc2_w->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }
        for (int i = 0; i < fc2_b->Size(); ++i) {
            fc2_b->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        if (use_gpu) {
            // fc1 = new LinearGPU(nullptr, nullptr, nullptr, input_size,
            // hidden_size); relu = new ReluGPU(nullptr, input_size); fc2 = new
            // LinearGPU(nullptr, nullptr, nullptr, hidden_size, output_size);
            // sigmoid = new SigmoidGPU(nullptr, hidden_size);
        } else {
            fc1 = new LinearCPU(input_size, hidden_size);
            relu = new ReluCPU();
            fc2 = new LinearCPU(hidden_size, output_size);
            sigmoid = new SigmoidCPU();
        }
    }

    std::shared_ptr<Tensor>
    Forward(const std::shared_ptr<Tensor> input) override final {
        auto fc1_out = fc1->Forward(input);
        auto relu_out = relu->Forward(fc1_out);
        auto fc2_out = fc2->Forward(relu_out);
        auto sigmoid_out = sigmoid->Forward(fc2_out);
        return sigmoid_out;
        // return
        // sigmoid->Forward(fc2->Forward(relu->Forward(fc1->Forward(input))));
    }

    void Backward(const std::shared_ptr<Tensor> loss, float lr) override final {
        // calcu grad, update weights params
        auto sig_diff = sigmoid->Backward(loss, lr);
        auto fc2_diff = fc2->Backward(sig_diff, lr);
        auto relu_diff = relu->Backward(sig_diff, lr);
        auto fc1_diff = fc1->Backward(relu_diff, lr);
    }

    ~SimpleNetwork() {
        delete fc1;
        delete relu;
        delete fc2;
        delete sigmoid;
    }
};
