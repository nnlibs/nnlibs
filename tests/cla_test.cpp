#include <cmath>
#include <iostream>
#include <utility>

#include "functional/functional.h"
#include "loss/cross_entropy.h"
#include "network/cla_network.h"
#include "tests/utils/image_ut.h"

std::pair<uint32_t, float> Predict(const std::shared_ptr<Tensor> &logits) {
    uint32_t max_idx = 0;
    float max_val = logits->data[0];
    for (int i = 1; i < logits->Size(); ++i) {
        if (logits->data[i] > max_val) {
            max_val = logits->data[i];
            max_idx = i;
        }
    }
    return std::make_pair(max_idx, max_val);
}

// int main(int argc, char const *argv[]) {
//     ClassifyNetwork clann;

//     // =========== train ============
//     int total_data = 1000;
//     std::vector<std::shared_ptr<Tensor>> inputs(total_data);
//     std::vector<std::shared_ptr<Tensor>> targets(total_data);
//     for (int i = 0; i < total_data; ++i) {
//         inputs[i] = std::make_shared<Tensor>(std::vector<int>{1, 1, 32, 32});
//         for (int j = 0; j < inputs[i]->Size(); ++j) {
//             inputs[i]->data[j] = (20 * (i + 1)) % 255;
//         }
//         // std::cout << "Input: " << inputs[i] << std::endl;
//         targets[i] = std::make_shared<Tensor>(std::vector<int>{1, 10});
//         targets[i]->data[i % 10] = 0.8;
//         targets[i]->data[(i + 1) % 10] = 0.1;
//         targets[i]->data[(i + 2) % 10] = 0.1;
//         // std::cout << "Target: " << targets[i] << std::endl;
//     }
//     int iter = 1;
//     for (int i = 0; i < iter; ++i) {
//         for (int j = 0; j < total_data; ++j) {
//             float iter_loss = 0.0f;
//             clann.ZeroGrad();
//             auto output = clann.Forward(inputs[j]);
//             // std::cout << "output " << j << " is " << output << std::endl;
//             iter_loss = F::MSELoss(output, targets[j]);
//             std::shared_ptr<Tensor> diff =
//                 std::make_shared<Tensor>(std::vector<int>{1, 10});
//             for (int k = 0; k < 10; ++k) {
//                 diff->data[i] =
//                     2 * (output->data[k] - targets[j]->data[k]) / 10;
//             }
//             clann.Backward(diff, 0.001);
//             std::cout << "---iter: " << i << ", loss: " << iter_loss
//                       << std::endl;
//         }
//     }

//     return 0;
// }

int main() {
    ClassifyNetwork clann;
    CrossEntropy ce;

    std::string train_file_prefix =
        "/home/aico/Downloads/images/cifar-10-batches-bin/";
    std::vector<std::string> train_files = {
        "data_batch_1.bin",
        // "data_batch_2.bin", "data_batch_3.bin",
        // "data_batch_4.bin", "data_batch_5.bin"
    };
    std::string test_file = "test_batch.bin";

    std::vector<std::vector<ClassifyImageData>> train_data_list;
    std::cout << "Loading train data..." << std::endl;
    for (auto &file : train_files) {
        auto image_data_list = LoadClassifyImageList(train_file_prefix + file);
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
                std::cout << "label: " << image_data_list[j].label
                          << " ,logits: " << logits << std::endl;
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
    std::cout << "Loading test data..." << std::endl;
    auto test_data_list = LoadClassifyImageList(train_file_prefix + test_file);
    uint32_t correct = 0;
    for (int i = 0; i < test_data_list.size(); ++i) {
        auto logits = clann.Forward(test_data_list[i].image);
        auto max_pair = Predict(logits);
        if (max_pair.first == test_data_list[i].label) {
            correct++;
            std::cout << "The [" << i << "] th image is OK, ID is ["
                      << max_pair.first << "], value is [" << max_pair.second
                      << "]" << std::endl;
        }
    }
    std::cout << "Accuracy: " << correct << "/" << test_data_list.size()
              << " = " << (float)correct / test_data_list.size() << std::endl;

    return 0;
}
