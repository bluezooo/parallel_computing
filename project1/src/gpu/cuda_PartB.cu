// CUDA implementation of smoothing a JPEG image using a filter
#include <iostream>
#include <cuda_runtime.h> // CUDA Header
#include "utils.hpp"

const int FILTER_SIZE = 9;
const float filter[9] = {
    1.0 / 9, 1.0 / 9, 1.0 / 9,
    1.0 / 9, 1.0 / 9, 1.0 / 9,
    1.0 / 9, 1.0 / 9, 1.0 / 9
};

// CUDA kernel functon
__global__ void smoothing(const unsigned char* input, unsigned char* output,
                          int input_width, int input_height, int num_channels, const float *filter)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (w > 0 && w < input_width - 1 && h > 0 && h < input_height - 1)    {//35.47ms

        int sum_r = 0, sum_g = 0, sum_b = 0;
        int output_index = (h * input_width + w) * num_channels;

        for (int i = 0; i < FILTER_SIZE; i++) {
            int index = ((i/3-1)* input_width + (i%3-1)) * num_channels + output_index;
            int channel_value_r = input[index];
            int channel_value_g = input[index + 1];
            int channel_value_b = input[index + 2];
            sum_r += channel_value_r * filter[i];
            sum_g += channel_value_g * filter[i];
            sum_b += channel_value_b * filter[i];
        }
        
        output[output_index] = static_cast<unsigned char>(sum_r);
        output[output_index + 1] = static_cast<unsigned char>(sum_g);
        output[output_index + 2] = static_cast<unsigned char>(sum_b);
    }
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_filter;
    cudaMalloc((void**)&d_input, 
                input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_output,
                input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_filter, 
                FILTER_SIZE * sizeof(float));
    // Copy input data from host to device
    cudaMemcpy(d_input, input_jpeg.buffer,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter,
                FILTER_SIZE * sizeof(float),
                cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // int blockSize = 512; // 256
    // int numBlocks =
    //     (input_jpeg.width * input_jpeg.height + blockSize - 1) / blockSize;
    // int numBlocks = (input_jpeg.width * input_jpeg.height) / blockSize + 1;

    dim3 numBlocks((input_jpeg.width / 32)+1, (input_jpeg.height / 16)+1);     //16*16: 35 ms
	dim3 blockSize(32, 16);//33.92ms
    cudaEventRecord(start, 0); // GPU start time
    smoothing<<<numBlocks, blockSize>>>(d_input, d_output, input_jpeg.width,
                                        input_jpeg.height,
                                        input_jpeg.num_channels, d_filter);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);

    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    // Write Image to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}