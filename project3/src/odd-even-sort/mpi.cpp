//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Odd-Even Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, int first, int last, MPI_Status* status) {
    /* Your code here!
       Implement parallel odd-even sort with MPI
    */


    if (taskid == MASTER) {



        int all_sorted = 0;
        while (!all_sorted){
            all_sorted =1;

            // Perform the odd phase
            for (int i = first+1; i <= last - 1; i += 2) {
                if (vec[i] > vec[i + 1]) {
                    std::swap(vec[i], vec[i + 1]);
                    all_sorted = 0;
                }
            }

            // Perform the even phase
            for (int i = first; i <= last - 1; i += 2) {
                if (vec[i] > vec[i + 1]) {
                    std::swap(vec[i], vec[i + 1]);
                    all_sorted = 0;
                }
            }




        }
    }
    else{
            int sorted = 1;
            // Perform the odd phase
            for (int i = first+1; i <= last - 1; i += 2) {
                if (vec[i] > vec[i + 1]) {
                    std::swap(vec[i], vec[i + 1]);
                    all_sorted = 0;
                }
            }

            // Perform the even phase
            for (int i = first; i <= last - 1; i += 2) {
                if (vec[i] > vec[i + 1]) {
                    std::swap(vec[i], vec[i + 1]);
                    all_sorted = 0;
                }
            }
            int * sorted_buf = &sorted;
            MPI_Send(sorted_buf, 1, MPI_INT , 0, 0, MPI_COMM_WORLD);

    }

}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
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

    const int size = atoi(argv[1]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    int i = 0;
    int size = vec.size();
    int length_per_task = (size + numtasks - 1 )/numtasks;
    std::vector<int> cuts(numtasks + 1, 0);
    while (i < numtasks-1){
        cuts[i+1] = cuts[i]+length_per_task;
        i++;
    }
    cuts[numtasks] = size;
    std::vector<int> index (numtasks, 0);
    std::vector<int> length (numtasks, length_per_task);
    length[numtasks-1] = cuts[numtasks] - cuts[numtasks-1];

    int first = cuts[taskid];
    int last = cuts[taskid+1]-1;

    auto start_time = std::chrono::high_resolution_clock::now();

    oddEvenSort(vec, numtasks, taskid, first, last, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}