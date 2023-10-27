
#include <iostream>
#include <cuda_runtime.h> // CUDA Header
#include "matrix.hpp"

// CUDA kernel functon


__global__ void matrix_multiply(const int * d_A, const int * d_B, int * d_C, int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N){
        const int * d_A_ptr = &d_A[row*K];
        const int * d_B_ptr = &d_B[col*N];
        int * d_C_ptr = &d_C[row*N];
        int sum = 0;
        for (int k = 0; k < K; k++) {
           sum += *(d_A_ptr+k) * *(d_B_ptr+k);
        }
        *(d_C_ptr + col) = sum;
    }

}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable \n"<<  "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n";
    }
    const std::string matrix1_path = argv[1];
    const std::string matrix2_path = argv[2];
    const std::string result_path = argv[3];
    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);
    if (matrix1.getCols() != matrix2.getRows()) {
        std::cerr<<"Matrix dimensions are not compatible for multiplication.";
    }

    int M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    // Get the transpose of matrix2
    Matrix matrix2_transpose = Matrix(N, K);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < K; ++j) {
            matrix2_transpose[i][j] = matrix2[j][i];
        }
    }

    // Allocate memory on host (CPU)
    Matrix result = Matrix(M, N);
    // Allocate memory on device (GPU)

    int *d_A, *d_B, *d_C;
    int sizeA = M *K * sizeof(int);
    int sizeB = K* N * sizeof(int);
    int sizeC = M *N * sizeof(int);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);


    for (int i = 0; i < M; ++i){
        cudaMemcpy(&d_A[i * K], matrix1[i], K * sizeof(int), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < N; ++i){
        cudaMemcpy(&d_B[i * K], matrix2_transpose[i], K * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 numBlocks((M + 7)/ 8, (N +7) / 8); 
	dim3 blockSize(8, 8);
    
    cudaEventRecord(start, 0); // GPU start time
    matrix_multiply<<<numBlocks, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);

    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    for (int i = 0; i < M; ++i){
        cudaMemcpy(result[i], &d_C[i*N], N * sizeof(int), cudaMemcpyDeviceToHost);
    }
    result.saveToFile(result_path);
    std::cout << "Output file to: " << result_path << std::endl;

    // Release allocated memory on device and host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // delete matrix1;
    // delete matrix2;
    // delete result;
    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}