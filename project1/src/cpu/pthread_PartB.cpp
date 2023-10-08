#include <iostream>
#include <chrono>
#include <pthread.h>
#include "utils.hpp"

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
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int start_height;
    int end_height;
};

// Function to convert RGB to Grayscale for a portion of the image
void* smoothing(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    
    for (int h = data->start_height; h < data->end_height; h++){
        // #pragma acc loop independent
        for (int w = 1; w < width - 1; w++){
            int sum_r = 0, sum_g = 0, sum_b = 0;
            int p = (h * width + w) * num_channels;

            for (int j = 0; j < FILTER_SIZE; j++) {
                int index = ((j/3-1)* width + (j%3-1)) * num_channels + p;
                unsigned char r = data->input_buffer[index];
                unsigned char g = data->input_buffer[index + 1];
                unsigned char b = data->input_buffer[index + 2];
                sum_r += r * filter[j];
                sum_g += g * filter[j];
                sum_b += b * filter[j];
            }
            data->output_buffer[p] = static_cast<unsigned char>(sum_r);
            data->output_buffer[p+1] = static_cast<unsigned char>(sum_g);
            data->output_buffer[p+2] = static_cast<unsigned char>(sum_b);
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
    
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    
    width = input_jpeg.width;
    height = input_jpeg.height;
    num_channels = input_jpeg.num_channels;

    auto start_time = std::chrono::high_resolution_clock::now();

    // int chunk_size = (input_jpeg.width-2) * (input_jpeg.height-2) / num_threads;
    int h_per_thread = (height- 2 + num_threads -1) / num_threads;
    #pragma acc loop
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input_buffer = input_jpeg.buffer;
        thread_data[i].output_buffer = filteredImage;
        thread_data[i].start_height = i * h_per_thread +1;
        thread_data[i].end_height = (i == num_threads - 1) ? (height-1):\
                                    (i + 1) * h_per_thread+1;
        pthread_create(&threads[i], nullptr, smoothing, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
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
