#!/bin/bash
#SBATCH -o ./Project2-Results.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1


CURRENT_DIR=$(pwd)

# Naive
echo "Naive Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/naive ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result_standard1.txt
echo ""

echo "Naive Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/naive ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result_standard2.txt
echo ""

# Memory Locality
echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/locality ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""

echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/locality ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""

# SIMD + Reordering
echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/simd ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""
echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/simd ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""

# OpenMP + SIMD + Reordering
echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/build/src/openmp $num_cores ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
  echo ""
#   python ${CURRENT_DIR}/com.py
done

echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/build/src/openmp $num_cores ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
  echo ""
done


### MPI + OpenMP + SIMD + Reordering
echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 32 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""

# # python ${CURRENT_DIR}/com.py

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 16 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 8 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 4 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 2 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""
# python ${CURRENT_DIR}/com.py
echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 1 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""



echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 32 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 16 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 8 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 4 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 2 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""

echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/build/src/mpi 1 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""





echo "------GPU----CUDA--------"

echo "CUDA"
srun -n 1 --gpus 1 ${CURRENT_DIR}/build/src/gpu/cuda ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""

echo "CUDA"
srun -n 1 --gpus 1 ${CURRENT_DIR}/build/src/gpu/cuda ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
echo ""

# python $(pwd)/com.py

# ################ profiling with per ##########################################
echo "------profiling----perf--------"
# Naive
echo "Naive Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/p1_naive_1024.data ${CURRENT_DIR}/build/src/naive ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result_standard1.txt
echo ""

# echo "Naive Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf1_naive_2048.data ${CURRENT_DIR}/build/src/naive ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result_standard2.txt
# echo ""

# Memory Locality
echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/p2_locality_1024.data ${CURRENT_DIR}/build/src/locality ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""

# echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf2_locality_2048.data ${CURRENT_DIR}/build/src/locality ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
# echo ""



# SIMD + Reordering
echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/p3_simd_1024.data ${CURRENT_DIR}/build/src/simd ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""
# echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf3_simd_2048.data ${CURRENT_DIR}/build/src/simd ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
# echo ""

# OpenMP + SIMD + Reordering
echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: ${num_cores}"
  srun -n 1 --cpus-per-task $num_cores perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/p4_openmp_${num_cores}threads_1024.data ${CURRENT_DIR}/build/src/openmp $num_cores ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
  echo ""
#   python ${CURRENT_DIR}/com.py
done

# echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# for num_cores in 1 2 4 8 16 32
# do
#   echo "Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf4_simd_openmp_2048.data ${CURRENT_DIR}/build/src/openmp $num_cores ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
#   echo ""
# done


# ### MPI + OpenMP + SIMD + Reordering
# echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# echo "Number of Processes: 1, Number of Threads: 32"
# srun -n 1 --cpus-per-task 32 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_openmp_simd_1024.data ${CURRENT_DIR}/build/src/mpi 32 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
# echo ""

# # python ${CURRENT_DIR}/com.py

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/p5_mpi_2tasks_1024.data ${CURRENT_DIR}/build/src/mpi 16 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
echo ""

# echo "Number of Processes: 4, Number of Threads: 8"
# srun -n 4 --cpus-per-task 8 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/p5_mpi_4tasks_1024.data ${CURRENT_DIR}/build/src/mpi 8 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
# echo ""

# echo "Number of Processes: 8, Number of Threads: 4"
# srun -n 8 --cpus-per-task 4 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_8tasks_1024.data ${CURRENT_DIR}/build/src/mpi 4 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
# echo ""

# echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/p5_                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                q         mpi_16tasks_1024.data ${CURRENT_DIR}/build/src/mpi 2 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
# echo ""
# python ${CURRENT_DIR}/com.py
# echo "Number of Processes: 32, Number of Threads: 1"
# srun -n 32 --cpus-per-task 1 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_openmp_simd_1024.data ${CURRENT_DIR}/build/src/mpi 1 ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt ${CURRENT_DIR}/build/result1.txt
# echo ""



# echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# echo "Number of Processes: 1, Number of Threads: 32"
# srun -n 1 --cpus-per-task 32 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_openmp_simd_2048.data ${CURRENT_DIR}/build/src/mpi 32 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
# echo ""

# echo "Number of Processes: 2, Number of Threads: 16"
# srun -n 2 --cpus-per-task 16 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_openmp_simd_2048.data ${CURRENT_DIR}/build/src/mpi 16 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
# echo ""

# echo "Number of Processes: 4, Number of Threads: 8"
# srun -n 4 --cpus-per-task 8 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_openmp_simd_2048.data ${CURRENT_DIR}/build/src/mpi 8 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
# echo ""

# echo "Number of Processes: 8, Number of Threads: 4"
# srun -n 8 --cpus-per-task 4 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_openmp_simd_2048.data ${CURRENT_DIR}/build/src/mpi 4 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
# echo ""

# echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_openmp_simd_2048.data ${CURRENT_DIR}/build/src/mpi 2 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
# echo ""

# echo "Number of Processes: 32, Number of Threads: 1"
# srun -n 32 --cpus-per-task 1 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./profiling/perf5_mpi_openmp_simd_2048.data ${CURRENT_DIR}/build/src/mpi 1 ${CURRENT_DIR}/matrices/matrix7.txt ${CURRENT_DIR}/matrices/matrix8.txt ${CURRENT_DIR}/build/result2.txt
# echo ""
