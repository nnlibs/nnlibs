#include <cmath>
#include <iostream>
#include <utility>

#include "loss/cross_entropy.h"
#include "network/cla_network.h"
#include "tests/utils/image_ut.h"

std::pair<uint8_t, float> Predict(const std::shared_ptr<Tensor> &logits) {
    uint8_t max_idx = 0;
    float max_val = logits->data[0];
    for (int i = 1; i < logits->Size(); ++i) {
        if (logits->data[i] > max_val) {
            max_val = logits->data[i];
            max_idx = static_cast<uint8_t>(i);
        }
    }
    return std::make_pair(max_idx, max_val);
}

int main() {
    ClassifyNetwork clann;
    CrossEntropy ce;

    std::string train_file_prefix =
        "/home/aico/Downloads/images/cifar-10-batches-bin/";
    std::vector<std::string> train_files = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin"};
    std::string test_file = "test_batch.bin";

    std::vector<std::vector<ClassifyImageData>> train_data_list;
    std::cout << "Loading train data..." << std::endl;
    for (auto &file : train_files) {
        auto image_data_list = LoadClassifyImageList(train_file_prefix + file);
        // image_data_list.resize(200);
        train_data_list.push_back(image_data_list);
    }
    std::cout << "Train start..." << std::endl;
    for (int ep = 1; ep <= 2; ++ep) {
        float ep_loss = 0.0f;
        for (auto image_data_list : train_data_list) {
            for (int j = 0; j < image_data_list.size(); ++j) {
                // clean grad
                clann.ZeroGrad();
                // run forward and get logits
                auto logits = clann.Forward(image_data_list[j].image);
                // init target
                auto target = std::make_shared<Tensor>(std::vector<int>{1});
                target->data[0] = image_data_list[j].label;
                // std::cout << "label: " << (uint32_t)image_data_list[j].label
                //           << " ,logits: " << logits << std::endl;
                ep_loss += ce.Forward(logits, target);
                auto loss_grad = ce.Backward(logits, target);
                clann.Backward(loss_grad, 0.001, 0.9);
                if (j % 200 == 199) {
                    std::cout << "Epoch: " << ep << ", iter: " << j
                              << ", loss: " << ep_loss / 200 << std::endl;
                    ep_loss = 0.0f;
                }
            }
        }
    }
    // clann.PrintDelay();
    std::cout << "Loading test data..." << std::endl;
    auto test_data_list = LoadClassifyImageList(train_file_prefix + test_file);
    uint32_t correct = 0;
    for (int i = 0; i < test_data_list.size(); ++i) {
        auto logits = clann.Forward(test_data_list[i].image);
        auto max_pair = Predict(logits);
        if (max_pair.first == test_data_list[i].label) {
            correct++;
            std::cout << "The [" << i << "] th image is OK, ID is ["
                      << (uint32_t)max_pair.first << "], value is ["
                      << max_pair.second << "]" << std::endl;
        }
    }
    std::cout << "Accuracy: " << correct << "/" << test_data_list.size()
              << " = " << (float)correct / test_data_list.size() << std::endl;

    return 0;
}
