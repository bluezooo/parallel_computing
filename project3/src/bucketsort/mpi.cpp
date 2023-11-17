//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0
void insertionSort(std::vector<int>& bucket) {
    for (int i = 1; i < bucket.size(); ++i) {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status) {
    /* Your code here!
       Implement parallel bucket sort with MPI
    */
    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());

    int range = max_val - min_val + 1;
    int small_bucket_size = range / num_buckets;
    int large_bucket_size = small_bucket_size + 1;
    int large_bucket_num = range - small_bucket_size * num_buckets;
    int boundary = min_val + large_bucket_num * large_bucket_size;




    int i = 0;
    int buckets_per_task = (num_buckets + numtasks - 1 )/numtasks;
    std::vector<int> cuts(numtasks + 1, 0);
    while (i < numtasks-1){
        cuts[i+1] = cuts[i]+buckets_per_task;
        i++;
    }
    cuts[numtasks] = num_buckets;


    // // Sort each bucket using insertion sort
    // int bucket_index = 0;
    // for (std::vector<int>& bucket : buckets) {
    //     if (bucket_index >= cuts[taskid] && bucket_index < cuts[taskid + 1]){
    //         insertionSort(bucket);
    //     }
    //     bucket_index ++;
    // }

    std::vector<std::vector<int>> buckets(num_buckets);
    // Pre-allocate space to avoid re-allocation
    for (std::vector<int>& bucket : buckets) {
        bucket.reserve(large_bucket_size);
    }

    // Place each element in the appropriate bucket
    for (int num : vec) {
        int index;
        if (num < boundary) {
            index = (num - min_val) / large_bucket_size;
        } else {
            index = large_bucket_num + (num - boundary) / small_bucket_size;
        }
        if (index >= num_buckets) {
            index = num_buckets - 1;
        }

        if (index < cuts[taskid+1] && index>= cuts[taskid] ){
            buckets[index].push_back(num);
        }
    }

    

    if (taskid == MASTER){
        for (int j = cuts[taskid]; j<cuts[taskid+1]; j++){
            insertionSort(buckets[j]);
        }

        int index = 0;
        for (int j = cuts[taskid]; j<cuts[taskid+1]; j++){
            for (int num : buckets[j]) {
                vec[index++] = num;
            }
        }

        int length = 0;
        int * length_buf = &length;
        for (int i = MASTER + 1; i < numtasks; i++) {
            int * start_pos = &vec[index];
            MPI_Recv(length_buf, 1, MPI_INT ,i, 0, MPI_COMM_WORLD, status);
            MPI_Recv(start_pos, length, MPI_INT, i, 0, MPI_COMM_WORLD, status);
            index+=length;
        }
    }
    else{
        int length = 0;
        for (int j = cuts[taskid]; j<cuts[taskid+1]; j++){
            length += buckets[j].size();
            insertionSort(buckets[j]);
        }

        std::vector<int> current_vec(length);
        int index = 0;
        for (int j = cuts[taskid]; j<cuts[taskid+1]; j++){
            for (int num : buckets[j]) {
                current_vec[index++] = num;
            }
        }
        int * length_ptr = &length;
        MPI_Send(length_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        int * current_vec_ptr = &current_vec[0];
        MPI_Send(current_vec_ptr, length, MPI_INT , 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size bucket_num\n"
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

    const int bucket_num = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}