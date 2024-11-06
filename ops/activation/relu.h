#pragma once

#include "operator.h"

class Relu : public Operator {
  public:
  protected:
    std::shared_ptr<Tensor> grad = nullptr;
};
