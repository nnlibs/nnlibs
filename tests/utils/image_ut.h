
#include "common/tensor.h"
#include "functional/functional.h"
#include <fstream>
#include <iostream>

struct ClassifyImageData {
    std::shared_ptr<Tensor> image;
    uint8_t label;
};

static std::vector<ClassifyImageData>
LoadClassifyImageList(const std::string &image_byte_data_file) {
    std::vector<ClassifyImageData> image_list;
    std::ifstream file(image_byte_data_file, std::ios::binary);
    if (!file.is_open()) {
        return image_list;
    }

    // int i = 0;
    // <1 x label><3072 x pixel>
    while (!file.eof()) {
        ClassifyImageData image_data;
        uint8_t label;
        file.read((char *)&label, sizeof(uint8_t));
        image_data.label = label;
        uint8_t *data = new uint8_t[3072];
        file.read((char *)data, 3072);

        image_data.image =
            std::make_shared<Tensor>(std::vector<int>({1, 3, 32, 32}));
        // convert to [0-1]
        for (int i = 0; i < 3072; i++) {
            image_data.image->data[i] = (float)(data[i]) / 255.0;
        }
        // normalize
        F::Normalize(image_data.image, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5});

        image_list.push_back(image_data);
        // ++i;
        // if (i == 3) {
        // std::cout << image_data.label << std::endl;
        // std::ofstream out("test_image.rgb_planar", std::ios::binary);
        // out.write(data, 3072);
        // out.close();
        //     for (int i = 0; i < 3072; i++) {
        //         std::cout << image_data.image->data[i] << " ";
        //     }
        // }

        delete[] data;
    }
    file.close();
    return image_list;
}
