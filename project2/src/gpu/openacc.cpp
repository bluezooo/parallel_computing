
#include <iostream>
#include <chrono>
#include <stdexcept>
#include "matrix.hpp"
// #include <openacc.h> // OpenACC Header

Matrix matrix_multiply(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

int main(int argc, char **argv)
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



    unsigned char *filteredImage = new unsigned char[width * height* num_channels];
    unsigned char *buffer = new unsigned char[width * height * num_channels];

    #pragma acc parallel loop
    for (int i = 0; i < width * height * num_channels; i++){
        buffer[i] = input_jpeg.buffer[i];
    }

    #pragma acc enter data copyin(filteredImage[0 : width * height* num_channels], \
                    buffer[0 : width * height * num_channels],\
                    filter[0: FILTER_SIZE], width, height, num_channels, FILTER_SIZE)
    #pragma acc update device(filteredImage[0 : width * height* num_channels], \
                    buffer[0 : width * height * num_channels],\
                    filter[0: FILTER_SIZE], width, height, num_channels, FILTER_SIZE)

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma acc parallel present(filteredImage[0 : width * height* num_channels], \
                                buffer[0 : width * height * num_channels],\
                                filter[0: FILTER_SIZE], width, height, num_channels, FILTER_SIZE) \
            num_gangs(1024)num_workers(1024)
            //(groups of threads) //256,1024,2048:35ms  //
            //num_workers //128: 34ms 512: 34ms 1024:34ms 2048:34ms 4096:34ms 8192:34ms
    {
        #pragma acc loop independent
        for (int h = 1; h < height - 1; h++)
        {

            #pragma acc loop independent
            for (int w = 1; w < width - 1; w++)
            {
                int sum_r = 0, sum_g = 0, sum_b = 0;
                int p = (h * width + w) * num_channels;

                for (int i = 0; i < FILTER_SIZE; i++) {
                    int index = ((i/3-1)* width + (i%3-1)) * num_channels + p;
                    unsigned char r = buffer[index];
                    unsigned char g = buffer[index + 1];
                    unsigned char b = buffer[index + 2];
                    sum_r += r * filter[i];
                    sum_g += g * filter[i];
                    sum_b += b * filter[i];
                }
                
                filteredImage[p] = static_cast<unsigned char>(sum_r);
                filteredImage[p + 1] = static_cast<unsigned char>(sum_g);
                filteredImage[p + 2] = static_cast<unsigned char>(sum_b);
            
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    #pragma acc update self(filteredImage[0 : width * height* num_channels])
                            //buffer[0 : width * height * num_channels])
    #pragma acc exit data copyout(filteredImage[0 : width * height* num_channels])

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
