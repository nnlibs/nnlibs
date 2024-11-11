#pragma once

#include "operator.h"

class Relu : public Operator {
  public:
    Relu(bool inplace = false) : inplace(inplace) {}

  protected:
    std::shared_ptr<Tensor> grad = nullptr;
    bool inplace = false;
};
