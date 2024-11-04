#pragma once

#include "common/tensor.h"

class CrossEntropy {
  public:
    CrossEntropy();

    float Forward(const std::shared_ptr<Tensor> logits,
                  const std::shared_ptr<Tensor> target);
    std::shared_ptr<Tensor> Backward(const std::shared_ptr<Tensor> logits,
                                     const std::shared_ptr<Tensor> target);

    std::shared_ptr<Tensor> Parameters();

    ~CrossEntropy();

  private:
    float softmax(const std::shared_ptr<Tensor> logits, int index);
};
