#pragma once

#include "operator.h"

class Sigmoid : public Operator {
 public:
 protected:
  std::shared_ptr<Tensor> grad = nullptr;
};