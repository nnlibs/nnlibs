#pragma once
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

// enum OpsType {
//     Add = 0,
//     Sub = 2,
//     Mul = 1,
//     Div = 3,
// };

class Tensor {
  public:
    Tensor(const std::vector<int> &shape) : shape(shape) {
        int total_size = std::accumulate(shape.begin(), shape.end(), 1,
                                         std::multiplies<int>());
        data.resize(total_size, 0.0f);
        strides = compute_strides(shape);
    }

    Tensor(const std::initializer_list<int> &shape) : shape(shape) {
        int total_size = std::accumulate(shape.begin(), shape.end(), 1,
                                         std::multiplies<int>());
        data.resize(total_size, 0.0f);
        strides = compute_strides(this->shape);
    }

    float &operator()(const std::vector<int> &indices) {
        assert(indices.size() == shape.size());
        int idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            assert(indices[i] < shape[i]);
            idx += strides[i] * indices[i];
        }
        return data[idx];
    }

    uint32_t GetIndexByIndices(const std::vector<int> &indices, int &idx) {
        if (indices.size() != shape.size()) {
            std::cerr << "indices size not match shape size" << std::endl;
            return 1;
        }
        idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                std::cout << "indices[" << i << "] = " << indices[i]
                          << " >= shape[" << i << "] = " << shape[i]
                          << std::endl;
                return 1;
            }
            idx += strides[i] * indices[i];
        }
        return 0;
    }

    uint32_t SetValue(const std::vector<int> &indices, float value) {
        if (indices.size() != shape.size()) {
            std::cerr << "indices size not match shape size" << std::endl;
            return 1;
        }
        int idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                std::cout << "indices[" << i << "] = " << indices[i]
                          << " >= shape[" << i << "] = " << shape[i]
                          << std::endl;
                return 1;
            }
            idx += strides[i] * indices[i];
        }
        data[idx] = value;
        return 0;
    }

    uint32_t GetValue(const std::vector<int> &indices, float &value) {
        if (indices.size() != shape.size()) {
            std::cerr << "indices size not match shape size" << std::endl;
            return 1;
        }
        int idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                std::cout << "indices[" << i << "] = " << indices[i]
                          << " >= shape[" << i << "] = " << shape[i]
                          << std::endl;
                return 1;
            }
            idx += strides[i] * indices[i];
        }
        value = data[idx];
        return 0;
    }

    // uint32_t OptsValue(const std::vector<int> &indices, float value,
    //                    OpsType type) {
    //     if (indices.size() != shape.size()) {
    //         std::cerr << "indices size not match shape size" << std::endl;
    //         return 1;
    //     }
    //     int idx = 0;
    //     for (size_t i = 0; i < indices.size(); ++i) {
    //         if (indices[i] >= shape[i]) {
    //             std::cout << "indices[" << i << "] = " << indices[i]
    //                       << " >= shape[" << i << "] = " << shape[i]
    //                       << std::endl;
    //             return 1;
    //         }
    //         idx += strides[i] * indices[i];
    //     }
    //     value = data[idx];
    //     return 0;
    // }

    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
        os << "Shape(";
        for (int i = 0; i < tensor.shape.size(); i++) {
            os << tensor.shape[i] << ",";
        }
        os << "),";
        os << "Tensor(";
        // shape is nchw, std::cout data format
        for (int i = 0; i < tensor.data.size(); i++) {
            os << tensor.data[i] << ",";
        }
        os << ")";
        return os;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const std::shared_ptr<Tensor> &tensor) {
        os << *tensor;
        return os;
    }

    void FillRandom(int max_value = 10) {
        for (int i = 0; i < data.size(); i++) {
            data[i] = rand() % max_value;
        }
    }

    Tensor Clone() {
        Tensor new_tensor(shape);
        new_tensor.data = data;
        new_tensor.strides = strides;
        return new_tensor;
    }

    float &operator[](int idx) { return data[idx]; }

    int Size() const { return data.size(); }

    void View(const std::vector<int> &new_shape) {
        assert(std::accumulate(new_shape.begin(), new_shape.end(), 1,
                               std::multiplies<int>()) == data.size());
        shape = new_shape;
        strides = compute_strides(new_shape);
    }

    // Tensor Reshape(const std::vector<int> &new_shape) const {
    //     assert(std::accumulate(new_shape.begin(), new_shape.end(), 1,
    //                            std::multiplies<int>()) == data.size());
    //     return Tensor(new_shape, data);
    // }

    ~Tensor() {}

    // 禁止拷贝，防止不必要的内存分配
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    // 允许移动语义
    Tensor(Tensor &&other) noexcept {
        data = std::move(other.data);
        shape = other.shape;
        strides = other.strides;
    }

  private:
    std::vector<int> compute_strides(const std::vector<int> &shape) {
        std::vector<int> strides(shape.size());
        strides.back() = 1;
        int stride = 1;
        for (int i = shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

  public:
    std::vector<float> data;

    // cv: (batch_size, channels, height, width)
    // nlp: (batch_size, sequence_length, embedding_dim)
    // fc: (batch_size, num_features)
    // transformer: (batch_size, num_heads, sequence_length,head_dim)
    std::vector<int> shape;
    std::vector<int> strides;
};
