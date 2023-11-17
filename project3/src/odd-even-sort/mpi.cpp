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

void oddEvenSort(std::vector<int>& vec, std::vector<int>& cuts, int numtasks, int taskid, int first, int last, int size, MPI_Status* status) {

    int recv_start, recv_end;
    int index = 0;

    while (index < size) {
        int start_mod = first % 2;
        int end_mod = last % 2;

        // Perform the odd phase
        if (index % 2 != 0) {
            for (int i = first + start_mod; i < last; i += 2) {
                if (vec[i] > vec[i + 1]) {
                    std::swap(vec[i], vec[i + 1]);
                }
            }

            if (start_mod == 0 && taskid != 0) {
                MPI_Sendrecv(&vec[first], 1, MPI_INT, taskid - 1, 2* taskid-1,
                             &recv_start, 1, MPI_INT, taskid - 1, 2*taskid -1, MPI_COMM_WORLD, status);
                if (recv_start > vec[first]) {
                    vec[first] = recv_start;
                }
            }
            if (end_mod != 0 && taskid != numtasks-1) {
                MPI_Sendrecv(&vec[last], 1, MPI_INT, taskid + 1, 2 * taskid+1,
                             &recv_end, 1, MPI_INT, taskid + 1, 2* taskid+1,MPI_COMM_WORLD, status);
                if (recv_end < vec[last]) {
                    vec[last] = recv_end;
                }
            }
        }

        // Perform the even phase
        else {
            for (int i = first + 1 - start_mod; i < last; i += 2) {
                if (vec[i] > vec[i + 1]) {
                    std::swap(vec[i], vec[i + 1]);
                }
            }
            
            if (start_mod != 0 && taskid != 0) {
                MPI_Sendrecv(&vec[first], 1, MPI_INT, taskid - 1, 2*taskid-1,
                             &recv_start, 1, MPI_INT, taskid - 1, 2*taskid-1, MPI_COMM_WORLD, status);
                if (recv_start > vec[first]) {
                    vec[first] = recv_start;
                }
            }
            if (end_mod == 0 && taskid != numtasks-1) {
                MPI_Sendrecv(&vec[last], 1, MPI_INT, taskid + 1, 2 * taskid+1,
                             &recv_end, 1, MPI_INT, taskid + 1, 2 * taskid+1, MPI_COMM_WORLD, status);
                if (recv_end < vec[last]) {
                    vec[last] = recv_end;
                }
            }
        }
        index ++;
    }
    
    if (taskid == MASTER) {
        for (int i = MASTER + 1; i < numtasks; i ++) {
            int * start_pos = &vec[cuts[i]];
            int length = last + 1 - first;
            MPI_Recv(start_pos, length, MPI_INT, i, 0, MPI_COMM_WORLD, status);
        }
    } else {
        int* start_pos = &vec[first];
        int length = last + 1 - first;
        MPI_Send(start_pos, length, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
    }
    // first the MPI
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
    // int size = vec.size();
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

    oddEvenSort(vec, cuts, numtasks, taskid, first, last, size, &status);

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