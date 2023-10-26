//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"


// // -------------debug-------------------------
// void printM256I(__m256i vector) {
//     int values[8];
//     _mm256_storeu_si256((__m256i*)&values, vector);
    
//     for (int i = 0; i < 8; i++) {
//         std::cout << (values[i])<< " ";
//     }
//     std::cout << std::endl;
// }

Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to Memory Locality and Cache Missing,
    // Further Applying SIMD
    for (size_t i = 0; i < M; ++i) {
        int * sum = result[i];
        for (size_t k = 0; k < K; ++k) {
            __m256i t1 = _mm256_set1_epi32((matrix1[i][k]));// printM256I(t1);
            const int *p1 = matrix2[k];
            for (size_t j = 0; j < N; j+=8) {
                __m256i sum_int = _mm256_loadu_si256((__m256i*)(sum+j));
                __m256i p1_int = _mm256_loadu_si256((__m256i*)(p1+j));
                __m256i product_int = _mm256_mullo_epi32(t1, p1_int);// printM256I(product_int);
                __m256i sum_added = _mm256_add_epi32(sum_int, product_int);
                int * pos = & sum[j];
                _mm256_storeu_si256((__m256i*)pos, sum_added);
            }
        }
    }
    return result;
}

// int a[] = {1;2;3;4;5;6;7;8;};
// int b[] = {0;0;0;0;0;0;0;0;};
// __m256i Va = _mm256_loadu_si256(a); // Va = a[0:8]; (CEAN notation [start:length])
// __m256i T = _mm256_add_epi32(Va, Va); / T = Va + Va
// _mm256_storeu_si256(b, T); // b[0:8] = T


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}