#pragma once

#include "operator.h"

class LayerNorm : public Operator {
 public:
  LayerNorm(const std::vector<int> &norm_shape) : norm_shape(norm_shape) {
    gamma = std::make_shared<Tensor>(norm_shape);  // 初始化缩放参数
    beta = std::make_shared<Tensor>(norm_shape);   // 初始化偏置参数

    // 初始化为1和0
    for (size_t i = 0; i < gamma->Size(); ++i) {
      gamma->data[i] = 1.0f;
      beta->data[i] = 0.0f;
    }
  }

  virtual ~LayerNorm() {}

 protected:
  std::vector<int> norm_shape;    // 归一化的形状（特征维度）
  std::shared_ptr<Tensor> gamma;  // 缩放参数
  std::shared_ptr<Tensor> beta;   // 偏置参数

  std::shared_ptr<Tensor> grad_gamma;
  std::shared_ptr<Tensor> grad_beta;
};
