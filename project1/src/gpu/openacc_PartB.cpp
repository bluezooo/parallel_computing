//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>

#include "utils.hpp"
// #include <openacc.h> // OpenACC Header

const int FILTER_SIZE = 9;
const float filter[9] = {
    1.0 / 9, 1.0 / 9, 1.0 / 9,
    1.0 / 9, 1.0 / 9, 1.0 / 9,
    1.0 / 9, 1.0 / 9, 1.0 / 9
};

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char *input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JPEGMeta input_jpeg = read_from_jpeg(input_filepath);
    // Computation: RGB to Gray
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    unsigned char *filteredImage = new unsigned char[width * height* num_channels];
    unsigned char *buffer = new unsigned char[width * height * num_channels];

    #pragma acc parallel loop
    for (int i = 0; i < width * height * num_channels; i++){
        buffer[i] = input_jpeg.buffer[i];
    }

    #pragma acc enter data copyin(filteredImage[0 : width * height* num_channels], \
                                    buffer[0 : width * height * num_channels])
    #pragma acc update device(filteredImage[0 : width * height* num_channels], \
                                    buffer[0 : width * height * num_channels])

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma acc parallel present(filteredImage[0 : width * height* num_channels], \
                                buffer[0 : width * height * num_channels]) \
            num_gangs(1024)//(groups of threads) //256,1024,2048:35ms  //
    {
        #pragma acc loop independent
        for (unsigned short int h = 1; h < height - 1; h++)
        {

            #pragma acc loop independent
            for (unsigned short int w = 1; w < width - 1; w++)
            {
                unsigned short int sum_r = 0, sum_g = 0, sum_b = 0;
                unsigned short int p = (h * width + w) * num_channels;

                // #pragma acc loop
                for (int i = 0; i < FILTER_SIZE; i++) {
                    unsigned short int index = ((i/3-1)* width + (i%3-1)) * num_channels + p;
                    unsigned short int r = buffer[index];
                    unsigned short int g = buffer[index + 1];
                    unsigned short int b = buffer[index + 2];
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
