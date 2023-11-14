//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"
#include <limits>

#define MASTER 0
#define TAG_GATHER 0

int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

void quickSortHelper(std::vector<int> &vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        quickSortHelper(vec, low, pivotIndex - 1);
        quickSortHelper(vec, pivotIndex + 1, high);
    }
}

void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status, int size) {
    /* Your code here!
       Implement parallel quick sort with MPI
    */
    int i = 0;
    // std::assert(taskid ==0);
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

    quickSortHelper(vec, cuts[taskid], cuts[taskid+1]-1);
    
    if (taskid == MASTER) {
        for (int i = MASTER + 1; i < numtasks; i++) {
            int* start_pos = &vec[cuts[i]];
            MPI_Recv(start_pos, length[i], MPI_INT, i, TAG_GATHER, MPI_COMM_WORLD, status);
        }
        //merge the k sorted arrays, each from cuts[i] to cuts[i+1] -1
        std::vector<int> vec_clone = vec;
        std::vector <int>::iterator it = vec.begin();
        while (it != vec.end()){
            int min = std::numeric_limits<int>::max();
            int arg_min;
            for (int j = 0; j < numtasks; j++){
                if (index[j] < length[j]){
                    int current = vec_clone[cuts[j]+index[j]];
                    if (current < min){
                        min = current;
                        arg_min = j;
                    }
                }
            }
            index[arg_min]++;
            * it = min;
            it++;
        }
    }
    else{
        int * ptr = &vec[cuts[taskid]];
        MPI_Send(ptr , length[taskid] , MPI_INT , MASTER , TAG_GATHER , MPI_COMM_WORLD);
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

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, numtasks, taskid, &status, size);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}