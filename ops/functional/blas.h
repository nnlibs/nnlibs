#pragma once
#include "cblas.h"
#include "common/tensor.h"
// #include <chrono>

namespace F {
// A: [M,K]
// B: [K,N]
// C: [M,N]
static std::shared_ptr<Tensor>
TensorMul(const std::shared_ptr<Tensor> A, const std::shared_ptr<Tensor> B,
          bool A_T = false,
          bool B_T = false, // A transpose, B transpose
          bool is_cpu = true, float alpha = 1.0f, float beta = 0.0f) {
    // auto start = std::chrono::high_resolution_clock::now();
    assert(A->shape.size() == 2);
    assert(B->shape.size() == 2);

    int M = A->shape[0];
    int N = B->shape[1];
    int K = A->shape[1];

    if (B_T) {
        N = B->shape[0];
    }
    if (A_T) {
        M = A->shape[1];
        K = A->shape[0];
    }

    int lda = K;
    int ldb = N;
    int ldc = N;

    if (B_T) {
        ldb = K;
    }

    if (A_T) {
        lda = M;
    }

    std::shared_ptr<Tensor> C =
        std::make_shared<Tensor>(std::vector<int>{M, N});

    if (is_cpu) {
        cblas_sgemm(CblasRowMajor, A_T ? CblasTrans : CblasNoTrans,
                    B_T ? CblasTrans : CblasNoTrans, M, N, K, alpha,
                    &A->data[0], lda, &B->data[0], ldb, beta, &C->data[0], ldc);
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration =
    //     std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "TensorMul time: " << duration.count() << "us" << std::endl;
    return C;
}
} // namespace F
