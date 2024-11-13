#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <vector>

void matrixMultiplyBlocked(const std::vector<float> &A,
                           const std::vector<float> &B, std::vector<float> &C,
                           int N, int blockSize) {
// 初始化输出矩阵 C
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

// 分块矩阵乘法
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                for (int ii = i; ii < i + blockSize && ii < N; ++ii) {
                    for (int jj = j; jj < j + blockSize && jj < N; ++jj) {
                        float sum = 0.0f;
                        for (int kk = k; kk < k + blockSize && kk < N; ++kk) {
                            sum += A[ii * N + kk] * B[kk * N + jj];
                        }
                        C[ii * N + jj] += sum;
                    }
                }
            }
        }
    }
}

int main() {
    int N = 512; // 定义矩阵大小 N x N
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    // 尝试不同的 BLOCK_SIZE
    for (int blockSize = 16; blockSize <= 128; blockSize *= 2) {
        auto start = std::chrono::high_resolution_clock::now();

        // 调用分块矩阵乘法
        matrixMultiplyBlocked(A, B, C, N, blockSize);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Block size: " << blockSize
                  << ", Time: " << duration.count() << " us" << std::endl;
    }

    return 0;
}
