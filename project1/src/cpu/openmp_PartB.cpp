
#include <iostream>
#include <chrono>
#include <omp.h>    // OpenMP header
#include "utils.hpp"


const int FILTER_SIZE = 9;
const float filter[9] = {
    1.0 / 9, 1.0 / 9, 1.0 / 9,
    1.0 / 9, 1.0 / 9, 1.0 / 9,
    1.0 / 9, 1.0 / 9, 1.0 / 9
};


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    
    // Separate R, G, B channels into three continuous arrays
    auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];

    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    auto f_R = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto f_G = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto f_B = new unsigned char[input_jpeg.width * input_jpeg.height];

    auto start_time = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(256);
    #pragma omp parallel for default(none) shared(rChannel, gChannel, bChannel, f_R, f_G, f_B, input_jpeg, filter)
    for (int h = 1; h < input_jpeg.height - 1; h++)
    {
        for (int w = 1; w < input_jpeg.width - 1; w++)
        {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            int p = (h * input_jpeg.width + w);

            for (int i = 0; i < FILTER_SIZE; i++) {
                int index = (i/3-1)* input_jpeg.width + (i%3-1)+ p;
                unsigned char r = rChannel[index];
                unsigned char g = gChannel[index];
                unsigned char b = bChannel[index];
                sum_r += r * filter[i];
                sum_g += g * filter[i];
                sum_b += b * filter[i];
            }

            f_R[p] = static_cast<unsigned char>(sum_r);
            f_G[p] = static_cast<unsigned char>(sum_g);
            f_B[p] = static_cast<unsigned char>(sum_b);
        
        }
    }

    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        filteredImage[i*input_jpeg.num_channels] = f_R[i];
        filteredImage[i*input_jpeg.num_channels+1] = f_G[i];
        filteredImage[i*input_jpeg.num_channels+2] = f_B[i];
    }

    // Save output JPEG GrayScale image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }

    // Release the allocated memory
    delete[] input_jpeg.buffer;
    delete[] rChannel;
    delete[] gChannel;
    delete[] bChannel;
    delete[] filteredImage;
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
