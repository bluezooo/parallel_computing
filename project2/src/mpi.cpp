//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0
#define TAG_GATHER 0

Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, \
                size_t start, size_t end, size_t M, size_t K, size_t N) {


    // size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(end-start, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to OpenMP, SIMD, Memory Locality and Cache Missing,
    // Further Applying MPI
    // Note:
    // You can change the argument of the function 
    // for your convenience of task division

    
#pragma omp parallel for default(none) shared(matrix1, matrix2, result, M, N, K, start, end)

    for (size_t i = start; i < end; i++) {
        int * sum = result[i-start];
        for (size_t k = 0; k < K; ++k) {
            __m256i t1 = _mm256_set1_epi32((matrix1[i][k]));
            const int *p1 = matrix2[k];
            for (size_t j = 0; j < N; j+=8) {
                __m256i sum_int = _mm256_loadu_si256((__m256i*)(sum+j));
                __m256i p1_int = _mm256_loadu_si256((__m256i*)(p1+j));
                __m256i product_int = _mm256_mullo_epi32(t1, p1_int);
                __m256i sum_added = _mm256_add_epi32(sum_int, product_int);
                int * pos = & sum[j];
                _mm256_storeu_si256((__m256i*)pos, sum_added);
            }
        }
    }

    return result;
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];
    const std::string matrix2_path = argv[3];
    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    int M = matrix1.getRows();
    int N = matrix2.getCols();
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }
    int K = matrix1.getCols();
    int h_per_task = (M + numtasks - 1 )/numtasks;
    int i = 0;
    std::vector<int> cuts(numtasks + 1, 0);
    while (i < numtasks-1){
        cuts[i+1] = cuts[i]+h_per_task;
        i++;
    }
    cuts[numtasks] = M;


    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        Matrix result = Matrix(M, N);
        Matrix temp_result = matrix_multiply_mpi(matrix1, matrix2, cuts[MASTER], cuts[MASTER+1], M, K, N);
#pragma omp parallel for default(none) shared(result, temp_result, N, cuts)
        for(int i = 0; i < cuts[1]; i++){
            for(int j = 0; j < N; j++){
                result[i][j] = temp_result[i][j];
            }
        }
        for (int i = MASTER + 1; i < numtasks; i++) {
            auto start_row = cuts[i];
            auto end_row = cuts[i + 1];
            for (size_t j = start_row; j < end_row; j++) {
                int* start_pos = result[j];
                MPI_Recv(start_pos, N, MPI_INT, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;
        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                   << " milliseconds" << std::endl;
    } else {
        // std::cout<<taskid<<std::endl;
        auto start = cuts[taskid];
        auto end = cuts[taskid + 1];
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, start, end, M, K, N);
        for(int i = cuts[taskid]; i < cuts[taskid+1]; i++){
            int * ptr = r
            esult[i-cuts[taskid]];
            MPI_Send(ptr , N , MPI_INT , MASTER , TAG_GATHER , MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}