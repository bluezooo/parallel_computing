## CSC4005 Distributed and Parallel Computing

## Project 3:  Sorting Algorithms with MPI

# Report

##### Yuhang, Wang

##### 120090246

## How to run my code

#### Programming Environment

<table>
  <tr>
    <th>Item</th>
    <th>Configuration / Version</th>
  </tr>
  <tr>
    <td align="center">System Type</td>
    <td>x86_64</td>
  </tr>
  <tr>
    <td align="center">Opearing System</td>
    <td>CentOS Linux release 7.5.1804</td>
  </tr>
  <tr>
    <td align="center">CPU</td>
    <td>
      Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
      <br/>
      20 Cores, 40 Threads
    </td>
  </tr>
  <tr>
    <td align="center">Memory</td>
    <td>100GB RAM</td>
  </tr>
  <tr>
    <td align="center">GPU</td>
    <td>one Nvidia Quadro RTX 4000 GPU card for each node</td>
  </tr>
  <tr>
    <td align="center">CUDA</td>
    <td>11.7</td>
  </tr>
  <tr>
    <td align="center">GCC</td>
    <td>Red Hat 7.3.1-5</td>
  </tr>
  <tr>
    <td align="center">CMake</td>
    <td>3.14.1</td>
  </tr>
</table>
#### Instruction for running my code

- Firstly, **`cd` into** the **`project3`* root folder. 
- Secondly, edit the file `src/sbatch.sh` , and change whatever you need.
- Finally, run the following code, and the correctness of sorting will show in the `Project3-Result.txt` file.

```apl
mkdir build && cd build
cmake ..
make -j32
cd ..
sbatch src/sbatch.sh
```



## Task 1: Process-Level Parallel Quick Sort with MPI

### Design and Implementation

- **Pivot Selection:** Each MPI process selects a pivot element from its assigned data block.
- **Partitioning:** Data is partitioned into two sublists - elements smaller and larger than the pivot.
- **Recursion:** Quick Sort is applied recursively to both sublists within each MPI process.
- **Combining:** After sorting, a sequential merge is performed to construct the final sorted list.

### Optimization and Discoveries

- Optimizations:
  - Efficient partitioning within each MPI process.
  - Minimizing communication during the sequential merge.

## Task 2: Process-Level Parallel Bucket Sort

### Design and Implementation

- **Bucket Creation:** MPI workers sort several buckets each, and sorted buckets are sent to the main worker.
- **Distribution:** Elements are distributed into buckets using a mapping function.
- **Sort Each Bucket:** Each worker sorts its assigned buckets using insertion sort.
- **Concatenation:** Main worker concatenates the sorted buckets to obtain the final result.

### Optimization and Discoveries

- Optimizations:
  - optimal number of buckets for performance.
  - concat  the 2-D dimensional buckets vector to a 1-D vector before sending and receiving.
- Discoveries:
  - The number of buckets significantly impacts performance, I compared and chose 800000.
  - Insertion sort is chosen for its simplicity and efficiency for small lists.

## Task 3: Process-Level Parallel Odd-Even Sort

### Design and Implementation

- **Initialization:** Dividing the list into odd and even indexed elements.
- **Iteration:** Odd and even phases of comparing and swapping elements within specific ranges.
- **Repeat Iterations:** Repeating iterations until no swaps occur, indicating the sorted list.
- **Finalize:** Communication mechanism to inform the master process about the sorted status.

### Optimization and Discoveries

- Optimizations:
  - Efficient communication for boundary comparisons in MPI.
  - Informing the master process about the sorted status at the end of each iteration.
- Discoveries:
  - MPI communication complexity increases when dealing with boundary elements.

## MY Execution Time

Performance measured as execution time in **milliseconds**.

The bucket number in the bucket sort need to be determined based on many experiments to decide the optimal parameter.

##### Here, I choose 800,000 out of 1,00,000,000 data as the number of buckets in Bucket Sort Algorithm

| Workers | QuickSort (MPI) | BucketSort (MPI) | Odd-Even-Sort (MPI) |
| ------- | --------------- | ---------------- | ------------------- |
| 1       | 13351           | 11375            | 41086               |
| 2       | 10587           | 8008             | 31772               |
| 4       | 6610            | 4695             | 19898               |
| 8       | 5811            | 3400             | 11695               |
| 16      | 7107            | 2823             | 6409                |
| 32      | 10732           | 2722             | 3401                |



## Profiling Records using perf

##### Quick Sort 8 cores (one core statement summary)

    72,769,706,648      cpu-cycles:u                                                
    72,582,441,914      cpu-cycles:u                                                
        17,055,580      cache-misses:u                                              
             3,723      page-faults:u                                               
    
      26.498208467 seconds time elapsed
    
      25.468490000 seconds user
       0.937771000 seconds sys

##### Bucket Sort 8 cores (one core statement summary)

    65,684,104,324      cpu-cycles:u                                                
        51,632,764      cache-misses:u                                              
           131,472      page-faults:u                                               
    
      24.239144620 seconds time elapsed
    
      23.014933000 seconds user
       1.150297000 seconds sys

##### Odd-Even Sort 8 cores (one core statement summary)

    32,780,063,487      cpu-cycles:u                                                
         2,256,873      cache-misses:u                                              
             2,416      page-faults:u                                               
    
      12.361552696 seconds time elapsed
    
      11.648105000 seconds user
       0.067837000 seconds sys

