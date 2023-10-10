#include <iostream>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include "utils.hpp"
#include <omp.h>

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
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }  


    // Prepross, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto new_reds = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto new_greens = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto new_blues = new unsigned char[input_jpeg.width * input_jpeg.height];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height -1; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i){
        filteredImage[i] = 0;
    }

    // Mask used for shuffling when store int32s to u_int8 arrays
    // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
    __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1,
                                    -1, -1, -1, -1);

    // // Using SIMD to accelerate the transformation
    auto start_time = std::chrono::high_resolution_clock::now();
    
    omp_set_num_threads(256);
    #pragma omp parallel for default(none) shared(reds, greens, blues, new_reds, new_greens, new_blues, input_jpeg, filter, shuffle)

    for (int i = input_jpeg.width+1; i < input_jpeg.width * (input_jpeg.height-1)-1; i+=8) {
        __m256 sumRed = _mm256_setzero_ps();
        __m256 sumGreen = _mm256_setzero_ps();
        __m256 sumBlue = _mm256_setzero_ps();

        for(int f = 0; f < FILTER_SIZE; f++){
            __m256 fil = _mm256_set1_ps(filter[f]);
            int index = i+ (f/3-1) * input_jpeg.width + (f%3-1);

            __m128i red_chars = _mm_loadu_si128((__m128i*) (reds+index));
            __m256i red_ints = _mm256_cvtepu8_epi32(red_chars);
            __m256 red_floats = _mm256_cvtepi32_ps(red_ints);
            __m256 red_results = _mm256_mul_ps(red_floats, fil);
            sumRed = _mm256_add_ps(red_results, sumRed);
            // printM256(sumRed);

            __m128i green_chars = _mm_loadu_si128((__m128i*) (greens +index));
            __m256i green_ints = _mm256_cvtepu8_epi32(green_chars);
            __m256 green_floats = _mm256_cvtepi32_ps(green_ints);
            __m256 green_results = _mm256_mul_ps(green_floats, fil);
            sumGreen = _mm256_add_ps(green_results, sumGreen);

            __m128i blue_chars = _mm_loadu_si128((__m128i*) (blues+index));
            __m256i blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            __m256 blue_floats = _mm256_cvtepi32_ps(blue_ints);
            __m256 blue_results = _mm256_mul_ps(blue_floats, fil);
            sumBlue = _mm256_add_ps(blue_results, sumBlue);
        }
        // printM256(sumRed);

        __m256i sumRed_ints =  _mm256_cvtps_epi32(sumRed); // Convert the float32 results to int32
        __m128i Red_low = _mm256_castsi256_si128(sumRed_ints);
        
        __m128i Red_high = _mm256_extracti128_si256(sumRed_ints, 1);
        __m128i Red_trans_low = _mm_shuffle_epi8(Red_low, shuffle);// shuffling int32s to u_int8s
        __m128i Red_trans_high = _mm_shuffle_epi8(Red_high, shuffle);// |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
        _mm_storeu_si128((__m128i*)(&new_reds[i]), Red_trans_low);
        _mm_storeu_si128((__m128i*)(&new_reds[i+4]), Red_trans_high);


        __m256i sumGreen_ints =  _mm256_cvtps_epi32(sumGreen);
        __m128i Green_low = _mm256_castsi256_si128(sumGreen_ints);
        __m128i Green_high = _mm256_extracti128_si256(sumGreen_ints, 1);
        __m128i Green_trans_low = _mm_shuffle_epi8(Green_low, shuffle);
        __m128i Green_trans_high = _mm_shuffle_epi8(Green_high, shuffle);
        _mm_storeu_si128((__m128i*)(&new_greens[i]), Green_trans_low);
        _mm_storeu_si128((__m128i*)(&new_greens[i+4]), Green_trans_high);


        __m256i sumBlue_ints =  _mm256_cvtps_epi32(sumBlue);
        __m128i Blue_low = _mm256_castsi256_si128(sumBlue_ints);
        __m128i Blue_high = _mm256_extracti128_si256(sumBlue_ints, 1);
        __m128i Blue_trans_low = _mm_shuffle_epi8(Blue_low, shuffle);
        __m128i Blue_trans_high = _mm_shuffle_epi8(Blue_high, shuffle);
        _mm_storeu_si128((__m128i*)(&new_blues[i]), Blue_trans_low);
        _mm_storeu_si128((__m128i*)(&new_blues[i+4]), Blue_trans_high);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout<< "ended"<< std::endl;
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    for (int j = 0; j < input_jpeg.width * input_jpeg.height -1; j++){
        filteredImage[input_jpeg.num_channels*j] = new_reds[j];
        filteredImage[input_jpeg.num_channels*j+1] = new_greens[j];
        filteredImage[input_jpeg.num_channels*j+2] = new_blues[j];
    }


    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}

