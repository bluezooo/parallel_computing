## CSC4005 Distributed and Parallel Computing

## Project 1:  Embarrassingly Parallel Programming

# Report

#### Yuhang, Wang

#### 120090246

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

- Firstly, `cd` into the `project1` root folder. 
- Secondly, download the testing images from https://github.com/tonyyxliu/CSC4005-2023Fall/tree/main/project1/images or using your own testing images. Then put the images under `image` folder.
- Thirdly, edit the file `src/scripts/sbatch_PartA.sh` and  `src/scripts/sbatch_PartB.sh`, especially the **filepaths** of the images.
- Finally, run the following code and the output will be in two `.txt` files.

```
mkdir build && cd build
cmake ..
make -j32
cd ..
sbatch ./src/scripts/sbatch_PartA.sh
sbatch ./src/scripts/sbatch_PartB.sh
```

## Part-A: RGB to Grayscale Image

#### About the six parallel programming languages

- **SIMD (Single Instruction, Multiple Data)**:
  - **Type**: Data-Level Parallelism (DLP)
  - SIMD does the same thing to a bunch of data at once. It's great for tasks where you want to apply the same operation to a lot of data, like doing math on arrays.
- **OpenMP**:
  - **Type**: Thread-Level Parallelism (TLP)
  - OpenMP makes your code run faster by using multiple threads. You tell it where to use threads, and it handles the details. It's like having several workers helping you out.
- **Pthread (POSIX Threads)**:
  - **Type**: Thread-Level Parallelism (TLP)
  -  You create and control threads yourself. It gives you precise control but requires more work.
  - Pthreads is like OpenMP's cousin, but it's easier to use.
- **MPI (Message Passing Interface)**:
  - **Type**: Data-Level Parallelism (DLP) and Task-Level Parallelism (TLP)
  - MPI is used when you have multiple computers working together. They talk to each other by passing messages. It's like a team of people working on different parts of a project.
- **CUDA (Compute Unified Device Architecture)**:
  - **Type**: Data-Level Parallelism (DLP), using GPU
  - CUDA supercharges your code by running it on NVIDIA GPUs. You write special GPU code (kernels) to do parallel tasks. It's like having a powerful graphics card do the heavy lifting.
- **OpenACC**:
  - **Type**: Data-Level Parallelism (DLP), using GPU
  - OpenACC makes GPU programming easier. You add special hints to your code, and it runs faster on GPUs. 
  - By using keyword `pragma`, it let programers help compiler find the loops.

#### MY Execution Time

Performance measured as execution time in milliseconds on a 20K JPEG image (19200 x 12995 = 250 million pixels)

| Number of Processes / Cores | Sequential | SIMD (AVX2) | MPI  | Pthread | OpenMP | CUDA   | OpenACC |
| --------------------------- | ---------- | ----------- | ---- | ------- | ------ | ------ | ------- |
| 1                           | 645        | 392         | 670  | 719     | 591    | 27.119 | 28      |
| 2                           | N/A        | N/A         | 786  | 645     |        | N/A    | N/A     |
| 4                           | N/A        | N/A         | 494  | 337     |        | N/A    | N/A     |
| 8                           | N/A        | N/A         | 344  | 179     |        | N/A    | N/A     |
| 16                          | N/A        | N/A         | 260  | 103     |        | N/A    | N/A     |
| 32                          | N/A        | N/A         | 219  | 68      |        | N/A    | N/A     |



## Part-B: Image Filtering (Soften with Equal Weight Filter)

The filter size and value is subject to user's change, and the execution time is proportional to the filter size.

#### My Execution Time

Performance measured as execution time in milliseconds on a 20K JPEG image (19200 x 12995 = 250 million pixels)

| Number of Processes / Cores | Sequential | SIMD (AVX2) | MPI   | Pthread | OpenMP | CUDA   | OpenACC |
| --------------------------- | ---------- | ----------- | ----- | ------- | ------ | ------ | ------- |
| 1                           | 10875      | 1518        | 13478 | 1514    | 1346   | 33.992 | 34      |
| 2                           | N/A        | N/A         | 11909 | 1336    | 1354   | N/A    | N/A     |
| 4                           | N/A        | N/A         | 6753  | 681     | 716    | N/A    | N/A     |
| 8                           | N/A        | N/A         | 3829  | 355     | 377    | N/A    | N/A     |
| 16                          | N/A        | N/A         | 2267  | 195     | 202    | N/A    | N/A     |
| 32                          | N/A        | N/A         | 1463  | 131     | 109    | N/A    | N/A     |



#### Combination of multiple programming language

##### Using SIMD in Pthread and OpenMP

- Since Pthread and OpenMP is Thread-Level Parallelism (TLP) and SIMD and SIMD is Data-Level Parallelism (DLP), they can be combined together to achieve a higher performance which takes far less running time.

- Actually, I think, in this project, SIMD can be used in every CPU acceleration, especially for the `for loop` acceleration, since it can calculate multiple pixels together.
- Remember to add `-mavx2` to CMakeLists.txt to compile pthread and openMP (SIMD combined) to enable AVX2 SIMD.

##### Using OpenMP in MPI

- Adding the following line  before the main `for loop`:

  ```cpp
  #pragma omp parallel for default(none) shared(rChannel, gChannel, bChannel, filteredImage, input_jpeg, filter, cuts)
  ```

- Since MPI divides the program execution into MASTER process and SLAVE processes, and combines them together. Therefore,  by using OpenMP within each process, openMP accelerates the running speed.
