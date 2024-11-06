#include "convolution/linear_cpu.h"
#include <memory>

int main() {
    int out_features = 5;
    int in_features = 10;
    auto weight =
        std::make_shared<Tensor>(std::vector<int>{out_features, in_features});
    auto bias = std::make_shared<Tensor>(std::vector<int>{out_features});

    weight->data = {1,  1, 0, 1, 1,  1, 0, 0,  1,  0, -1, 0, 1, 1, -1, 0, 1,
                    1,  1, 0, 1, -1, 0, 1, -1, 1,  1, 1,  1, 0, 2, 0,  1, 2,
                    -1, 1, 1, 1, 1,  0, 0, 0,  -1, 2, 1,  2, 1, 1, 1,  0};

    LinearCPU linear(in_features, out_features, weight, bias);

    auto input = std::make_shared<Tensor>(std::vector<int>{1, in_features});
    for (int i = 0; i < in_features; i++) {
        input->data[i] = i + 1;
    }

    auto output = linear.Forward(input);
    std::cout << "Output: " << output << std::endl;

    auto grad_output =
        std::make_shared<Tensor>(std::vector<int>{1, out_features});
    grad_output->data = {1, 0, 0, 1, 1};

    auto grad_input = linear.Backward(grad_output, 0.001);
    std::cout << "Grad input: " << grad_input << std::endl;
    return 0;
}
