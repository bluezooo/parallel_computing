#include <iostream>
#include <chrono>
#include <pthread.h>
#include "utils.hpp"
#include <immintrin.h>

const int FILTER_SIZE = 9;
const float filter[9] = {
    1.0 / 9, 1.0 / 9, 1.0 / 9,
    1.0 / 9, 1.0 / 9, 1.0 / 9,
    1.0 / 9, 1.0 / 9, 1.0 / 9
};
int width;
int height;
int num_channels;

// Structure to pass data to each thread
struct ThreadData {
    unsigned char* reds;
    unsigned char* greens;
    unsigned char* blues;
    unsigned char* new_reds;
    unsigned char* new_greens;
    unsigned char* new_blues;
    int start_height;
    int end_height;
};

__m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                -1, -1, -1, -1, 
                                -1, -1, -1, -1,
                                -1, -1, -1, -1);
                                
// Function to convert RGB to Grayscale for a portion of the image
void* smoothing(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    // #pragma omp parallel for default(none) shared(data, input_jpeg, filter)
    // for (int h = data->start_height; h < data->end_height; h++){
    //     for (int w = 1; w < width - 1; w++){
    //         int sum_r = 0, sum_g = 0, sum_b = 0;
    //         int p = (h * width + w) * num_channels;

    //         for (int j = 0; j < FILTER_SIZE; j++) {
    //             int index = ((j/3-1)* width + (j%3-1)) * num_channels + p;
    //             unsigned char r = data->input_buffer[index];
    //             unsigned char g = data->input_buffer[index + 1];
    //             unsigned char b = data->input_buffer[index + 2];
    //             sum_r += r * filter[j];
    //             sum_g += g * filter[j];
    //             sum_b += b * filter[j];
    //         }
    //         data->output_buffer[p] = static_cast<unsigned char>(sum_r);
    //         data->output_buffer[p+1] = static_cast<unsigned char>(sum_g);
    //         data->output_buffer[p+2] = static_cast<unsigned char>(sum_b);
    //     }
    // }
    for (int h = data->start_height; h < data->end_height; h++){
        for (int w = 1; w < width - 1; w+=8){

            __m256 sumRed = _mm256_setzero_ps();
            __m256 sumGreen = _mm256_setzero_ps();
            __m256 sumBlue = _mm256_setzero_ps();
            int i = (h * width + w);
            for(int f = 0; f < FILTER_SIZE; f++){
                __m256 fil = _mm256_set1_ps(filter[f]);
                int index = i+ (f/3-1) * width + (f%3-1);

                __m128i red_chars = _mm_loadu_si128((__m128i*) (data->reds+index));
                __m256i red_ints = _mm256_cvtepu8_epi32(red_chars);
                __m256 red_floats = _mm256_cvtepi32_ps(red_ints);
                __m256 red_results = _mm256_mul_ps(red_floats, fil);
                sumRed = _mm256_add_ps(red_results, sumRed);

                __m128i green_chars = _mm_loadu_si128((__m128i*) (data->greens +index));
                __m256i green_ints = _mm256_cvtepu8_epi32(green_chars);
                __m256 green_floats = _mm256_cvtepi32_ps(green_ints);
                __m256 green_results = _mm256_mul_ps(green_floats, fil);
                sumGreen = _mm256_add_ps(green_results, sumGreen);

                __m128i blue_chars = _mm_loadu_si128((__m128i*) (data->blues+index));
                __m256i blue_ints = _mm256_cvtepu8_epi32(blue_chars);
                __m256 blue_floats = _mm256_cvtepi32_ps(blue_ints);
                __m256 blue_results = _mm256_mul_ps(blue_floats, fil);
                sumBlue = _mm256_add_ps(blue_results, sumBlue);
            }

            __m256i sumRed_ints =  _mm256_cvtps_epi32(sumRed); // Convert the float32 results to int32
            __m128i Red_low = _mm256_castsi256_si128(sumRed_ints);
            
            __m128i Red_high = _mm256_extracti128_si256(sumRed_ints, 1);
            __m128i Red_trans_low = _mm_shuffle_epi8(Red_low, shuffle);// shuffling int32s to u_int8s
            __m128i Red_trans_high = _mm_shuffle_epi8(Red_high, shuffle);// |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
            _mm_storeu_si128((__m128i*)(&data->new_reds[i]), Red_trans_low);
            _mm_storeu_si128((__m128i*)(&data->new_reds[i+4]), Red_trans_high);


            __m256i sumGreen_ints =  _mm256_cvtps_epi32(sumGreen);
            __m128i Green_low = _mm256_castsi256_si128(sumGreen_ints);
            __m128i Green_high = _mm256_extracti128_si256(sumGreen_ints, 1);
            __m128i Green_trans_low = _mm_shuffle_epi8(Green_low, shuffle);
            __m128i Green_trans_high = _mm_shuffle_epi8(Green_high, shuffle);
            _mm_storeu_si128((__m128i*)(&data->new_greens[i]), Green_trans_low);
            _mm_storeu_si128((__m128i*)(&data->new_greens[i+4]), Green_trans_high);


            __m256i sumBlue_ints =  _mm256_cvtps_epi32(sumBlue);
            __m128i Blue_low = _mm256_castsi256_si128(sumBlue_ints);
            __m128i Blue_high = _mm256_extracti128_si256(sumBlue_ints, 1);
            __m128i Blue_trans_low = _mm_shuffle_epi8(Blue_low, shuffle);
            __m128i Blue_trans_high = _mm_shuffle_epi8(Blue_high, shuffle);
            _mm_storeu_si128((__m128i*)(&data->new_blues[i]), Blue_trans_low);
            _mm_storeu_si128((__m128i*)(&data->new_blues[i+4]), Blue_trans_high);
        }
    }
    return nullptr;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }

    int num_threads = std::stoi(argv[3]); // User-specified thread count

    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);

    // Computation: RGB to Gray
    unsigned char *filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
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

    
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    
    width = input_jpeg.width;
    height = input_jpeg.height;
    num_channels = input_jpeg.num_channels;

    int h_per_thread = (height- 2 + num_threads -1) / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].reds = reds;
        thread_data[i].greens = greens;
        thread_data[i].blues = blues;
        thread_data[i].new_reds = new_reds;
        thread_data[i].new_greens = new_greens;
        thread_data[i].new_blues = new_blues;
        thread_data[i].start_height = i * h_per_thread +1;
        thread_data[i].end_height = (i == num_threads - 1) ? (height-1):(i + 1) * h_per_thread+1;
    }

    
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], nullptr, smoothing, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
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
