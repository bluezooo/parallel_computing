
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>    // OpenMP header

#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

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

    // Read JPEG File
    const char * input_filepath = argv[1];
    // std::cout << "Input file from: " << input_filepath << "\n";
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

    // auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    // auto f_R = new unsigned char[input_jpeg.width * input_jpeg.height];
    // auto f_G = new unsigned char[input_jpeg.width * input_jpeg.height];
    // auto f_B = new unsigned char[input_jpeg.width * input_jpeg.height];




    // Divide the task
    // For example, there are 11 pixels and 3 tasks, 
    // we try to divide to 4 4 3 instead of 3 3 5

    int total_pixel_num =(input_jpeg.width) * (input_jpeg.height-2);
    int pixel_num_per_task = (total_pixel_num +(numtasks-1))/ numtasks;
    std::vector<int> cuts(numtasks + 1, 0);

    int i = 0;
    cuts[0] = input_jpeg.width;
    while (i < numtasks-1){
        cuts[i+1] = cuts[i] + pixel_num_per_task;
        i++;
    }
    cuts[i+1] = (input_jpeg.width) * (input_jpeg.height-1);
    // // std::cout<<numtasks<<std::endl;
    // // for (int i = 0; i < numtasks;i++){
    // //     std::cout<<cuts[i]<<std::endl;
    // // }

    // The tasks for the master executor
    // 1. Transform the first division of the RGB contents to the Gray contents
    // 2. Receive the transformed Gray contents from slave executors
    // 3. Write the Gray contents to the JPEG File
    // std::cout<<taskid<<std::endl;
    if (taskid == MASTER) {
        // Transform the first division of RGB Contents to the gray contents

        auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
        
        auto start_time = std::chrono::high_resolution_clock::now();
            
        #pragma omp parallel for default(none) shared(rChannel, gChannel, bChannel, filteredImage, input_jpeg, filter, cuts)

        for (int i = cuts[MASTER]; i < cuts[MASTER + 1]; i++) {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            for (int f = 0; f < FILTER_SIZE; f++) {
                int index = ((f/3-1)* input_jpeg.width + (f%3-1)+i);
                unsigned char r = rChannel[index];
                unsigned char g = gChannel[index];
                unsigned char b = bChannel[index];
                sum_r += r * filter[f];
                sum_g += g * filter[f];
                sum_b += b * filter[f];
            }

            filteredImage[i*input_jpeg.num_channels] = static_cast<unsigned char>(sum_r);
            filteredImage[i*input_jpeg.num_channels+1] = static_cast<unsigned char>(sum_g);
            filteredImage[i*input_jpeg.num_channels+2] = static_cast<unsigned char>(sum_b);
        }

        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filteredImage + cuts[i]*input_jpeg.num_channels; //gray-》ss
            int length = (cuts[i+1] - cuts[i])*input_jpeg.num_channels;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        //     filteredImage[i*input_jpeg.num_channels] = f_R[i];
        //     filteredImage[i*input_jpeg.num_channels+1] = f_G[i];
        //     filteredImage[i*input_jpeg.num_channels+2] = f_B[i];
        // }
        // Save the Gray Image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        // Release the memory
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    } 
    // The tasks for the slave executor
    // 1. Transform the RGB contents to the Gray contents
    // 2. Send the transformed Gray contents back to the master executor
    else {
        // Transform the RGB Contents to the gray contents
        int length = (cuts[taskid + 1] - cuts[taskid]) *input_jpeg.num_channels; 
        auto filteredImage = new unsigned char[length];

        #pragma omp parallel for default(none) shared(rChannel, gChannel, bChannel, filteredImage, input_jpeg, filter, cuts)

        for (int i = 0; i < cuts[taskid + 1]-cuts[taskid]; i++) {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            for (int f = 0; f < FILTER_SIZE; f++) {
                int index = ((f/3-1)* input_jpeg.width + (f%3-1)+ i+cuts[taskid]);
                unsigned char r = rChannel[index];
                unsigned char g = gChannel[index];
                unsigned char b = bChannel[index];
                sum_r += r * filter[f];
                sum_g += g * filter[f];
                sum_b += b * filter[f];
            }

            filteredImage[i*input_jpeg.num_channels] = static_cast<unsigned char>(sum_r);
            filteredImage[i*input_jpeg.num_channels+1] = static_cast<unsigned char>(sum_g);
            filteredImage[i*input_jpeg.num_channels+2] = static_cast<unsigned char>(sum_b);
        }

        // Send the gray image back to the master
        MPI_Send(filteredImage, length, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        
        // Release the memory
        delete[] filteredImage;
    }

    MPI_Finalize();
    return 0;
}
