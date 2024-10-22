#include <cmath>
#include <iostream>

#include "network/simple_network.h"

float CalcuLoss(SimpleNetwork& nn,
                const std::vector<std::shared_ptr<Tensor>> inputs,
                const std::vector<float>& targets) {
  float loss = 0.0f;
  for (int i = 0; i < inputs.size(); ++i) {
    loss += std::pow((nn.Forward(inputs[i]))->data[0] - targets[i], 2);
  }
  return loss / inputs.size();
}

void UpdateParams(SimpleNetwork& nn, float lr,
                  const std::vector<std::shared_ptr<Tensor>>& inputs,
                  const std::vector<float>& targets) {
  int total_data = inputs.size();
  std::shared_ptr<Tensor> diff =
      std::make_shared<Tensor>(std::vector<int>{total_data});
  for (int i = 0; i < total_data; ++i) {
    diff->data[i] =
        2 * (nn.Forward(inputs[i])->data[0] - targets[i]) / total_data;
  }
  nn.Backward(diff, lr);
}

int main(int argc, char const* argv[]) {
  SimpleNetwork nn(5, 10, 1);
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<int>{5});
  std::srand(static_cast<unsigned int>(std::time(0)));
  for (int i = 0; i < input->Size(); ++i) {
    input->data[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }
  // ============ infer ============
  auto output = nn.Forward(input);
  // 输出结果
  std::cout << "Output: ";
  for (size_t i = 0; i < output->Size(); ++i) {
    std::cout << output->data[i] << " ";
  }
  std::cout << std::endl;
  // ==============================

  // =========== train ============
  // dataset
  int total_data = 1;
  std::vector<std::shared_ptr<Tensor>> inputs(total_data);
  std::vector<float> targets(total_data);
  for (int i = 0; i < total_data; ++i) {
    inputs[i] = std::make_shared<Tensor>(std::vector<int>{5});
    for (int j = 0; j < inputs[i]->Size(); ++j) {
      inputs[i]->data[j] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    targets[i] = inputs[i]->data[0];
  }
  // iter train
  int iter = 10;
  for (int i = 0; i < iter; ++i) {
    std::cout << "Iter: " << i << ", loss: ";
    float loss = CalcuLoss(nn, inputs, targets);
    std::cout << loss << std::endl;

    std::cout << "update params...\n";
    UpdateParams(nn, 0.01, inputs, targets);
  }

  return 0;
}
