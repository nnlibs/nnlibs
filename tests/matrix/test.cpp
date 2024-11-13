// clang-format off
#include "common/tensor.h"
#include <chrono>
#include <iostream>
#include "cblas.h"
// clang-format on

int main() {
    Tensor t1({2, 30, 40});
    Tensor t2({2, 40, 50});
    Tensor t3({2, 30, 50});
    // t1 random data

    t1.FillRandom();
    t2.FillRandom();

    int N = 30;
    int M = 50;
    int K = 40;
    // ijp 28862 us
    // ipj 29761 us
    // jpi 26343 us
    // jip 31767 us
    // pij 30800 us
    // pji 25363 us

    auto start = std::chrono::high_resolution_clock::now();
    // use cblas to calculate
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1.0,
                (double *)&t1.data[0], K, (double *)&t2.data[0], M, 0.0,
                (double *)&t3.data[0], M);

    // for (int p = 0; p < K; p++) {         // p
    //     for (int j = 0; j < M; j++) {     // j
    //         for (int i = 0; i < N; i++) { // i
    //             t3({i, j}) += t1({i, p}) * t2({p, j});
    //         }
    //     }
    // }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time: " << duration.count() << " us" << std::endl;
}
