
#include <iostream>
#include <cuda_runtime.h> // CUDA Header
#include "matrix.hpp"

// CUDA kernel functon


__global__ Matrix matrix_multiply(const Matrix& matrix1_in, const Matrix& matrix2_in, const Matrix& result_out,  int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (size_t k = 0; k < K; ++k) {
        result_out[row][col] += matrix1_in[row][k] * matrix2_in[k][col];
    }
    return result;
}

int main(int argc, char** argv)
{
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
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();


    // Allocate memory on host (CPU)
    Matrix result = Matrix(M, N);
    // Allocate memory on device (GPU)
    int * matrix1_in;
    int * matrix2_in;
    int * result_out;
    int * K_in;

    cudaMalloc((void**)&matrix1_in, M*K * sizeof(int));
    cudaMalloc((void**)&matrix2_in, K*N * sizeof(int));
    cudaMalloc((void**)&K_in, sizeof(int));

    cudaMemcpy(matrix1_in, matrix1, M*K * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(matrix2_in, matrix2, K*N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(K_in, K, sizeof(int),cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 numBlocks((M + 15)/ 16, (N +15) / 16); 
	dim3 blockSize(16, 16);
    
    cudaEventRecord(start, 0); // GPU start time
    matrix_multiply<<<numBlocks, blockSize>>>(matrix1_in, matrix2_in, result_out, K_in);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);

    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(result, result_out, M*N * sizeof(int),cudaMemcpyDeviceToHost);
    
    result.saveToFile(result_path);
    std::cout << "Output file to: " << result_path << std::endl;

    // Release allocated memory on device and host
    cudaFree(matrix1_in);
    cudaFree(matrix2_in);
    cudaFree(result_out);
    // delete matrix1;
    // delete matrix2;
    // delete result;
    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}