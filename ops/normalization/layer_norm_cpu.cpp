#include "layer_norm_cpu.h"

#include <cmath>
LayerNormCPU::LayerNormCPU(const std::vector<int>& norm_shape)
    : LayerNorm(norm_shape) {}

std::shared_ptr<Tensor> LayerNormCPU::Forward(std::shared_ptr<Tensor> input) {
  int batch = input->shape[0];  // 批量大小
  // 计算均值和方差
  std::vector<float> batch_means(batch, 0.0f);
  std::vector<float> batch_vars(batch, 0.0f);

  int layer_size = input->Size() / batch;
  for (int i = 0; i < batch; ++i) {
    batch_means[i] = 0.0f;
    batch_vars[i] = 0.0f;

    for (int j = 0; j < layer_size; ++j) {
      batch_means[i] += input->data[i * layer_size + j];
    }
    batch_means[i] /= layer_size;

    for (int j = 0; j < layer_size; ++j) {
      float diff = input->data[i * layer_size + j] - batch_means[i];
      batch_vars[i] += diff * diff;
    }
    batch_vars[i] /= layer_size;
  }

  // 归一化
  std::shared_ptr<Tensor> output = std::make_shared<Tensor>(input->shape);
  for (size_t i = 0; i < batch; ++i) {
    for (size_t j = 0; j < layer_size; ++j) {
      output->data[i * layer_size + j] =
          gamma->data[i] * (input->data[i * layer_size + j] - batch_means[i]) /
              std::sqrt(batch_vars[i] + 1e-5) +
          beta->data[i];
    }
  }

  return output;
}

std::shared_ptr<Tensor> LayerNormCPU::Backward(
    std::shared_ptr<Tensor> grad_output, float lr) {}

LayerNormCPU::~LayerNormCPU() {}
