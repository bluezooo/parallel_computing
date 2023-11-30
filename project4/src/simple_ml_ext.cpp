#include "simple_ml_ext.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstring>
DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim)
{
    images_matrix = new float[images_num * input_dim];
    labels_array = new unsigned char[images_num];
}

DataSet::~DataSet()
{
    delete[] images_matrix;
    delete[] labels_array;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename)
{
    std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
    uint32_t magic_num, images_num, rows_num, cols_num;

    images_file.read(reinterpret_cast<char *>(&magic_num), 4);
    labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

    images_file.read(reinterpret_cast<char *>(&images_num), 4);
    labels_file.read(reinterpret_cast<char *>(&images_num), 4);
    images_num = swap_endian(images_num);

    images_file.read(reinterpret_cast<char *>(&rows_num), 4);
    rows_num = swap_endian(rows_num);
    images_file.read(reinterpret_cast<char *>(&cols_num), 4);
    cols_num = swap_endian(cols_num);

    DataSet *dataset = new DataSet(images_num, rows_num * cols_num);

    labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);
    unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
    images_file.read(reinterpret_cast<char *>(pixels), images_num * rows_num * cols_num);
    for (size_t i = 0; i < images_num * rows_num * cols_num; i++)
    {
        dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;
    }

    delete[] pixels;

    return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(float *A, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * k
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * n
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t M, size_t K, size_t N)
{

    memset(C, 0, M*N*sizeof(float));

    // for (size_t i = 0; i < M; ++i) {
    //     float * sum = &C[i*N];
    //     for (size_t k = 0; k < K; ++k) {
    //         float t1 = A[i*K+k];
    //         const float *p1 = &B[k*N];
    //         for (size_t j = 0; j < N; ++j) {
    //             sum[j] += p1[j] * t1;
    //         }
    //     }
    // }

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                C[i*N+j] += B[k*N+j] * A[i*K+k];
            }
        }
    }

    // std::cout<<"e";
    // size_t alignedN = (N / 8) * 8;

    // for (size_t i = 0; i < M; ++i)
    // {
    //     std::cout<<"d";
    //     float *sum = &C[i * N];
    //     std::cout<<"c";
    //     for (size_t k = 0; k < K; ++k)
    //     {
    //         float t1 = A[i * K + k];
    //         const float *p1 = &B[k * N];

    //         // Move t1 calculation outside of the loop
    //         __m256 t1_vec = _mm256_set1_ps(t1);
    //         std::cout<<"b";
    //         // Perform AVX2-based vectorized multiplication and addition
    //         for (size_t j = 0; j < alignedN; j += 8)
    //         {
    //                             // Check pointers and references before using them
    //             if (sum == nullptr || p1 == nullptr) {
    //                 std::cout<<"null ptr"<<std::endl;
    //                 return;
    //             }
    //             __m256 p1_vec = _mm256_loadu_ps(&p1[j]);
    //             __m256 sum_vec = _mm256_load_ps(&sum[j]);  // Use aligned load for sum
    //             __m256 add_result = _mm256_add_ps(_mm256_mul_ps(t1_vec, p1_vec), sum_vec);
    //             std::cout<<"a";
    //             // Use aligned store for sum
    //             // _mm256_store_ps(reinterpret_cast<float*>(&sum[j]), add_result);
    //             float * pos = &sum[j];
    //             _mm256_store_ps(pos, add_result);
    //         }

    //         // Handle the remaining elements (if any) using scalar operations
    //         for (size_t j = alignedN; j < N; ++j)
    //         {
    //                 if (sum == nullptr || p1 == nullptr) {
    //                     std::cout<<"null ptr"<<std::endl;
    //                 return;
    //             }
    //             sum[j] += p1[j] * t1;
    //         }
    //     }
    // }
    // END YOUR CODE
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size K * M
 *     B (const float*): Matrix of size K * N
 *     C (float*): Matrix of size M * N
 **/
void matrix_dot_trans(const float *A, const float *B, float *C, size_t K, size_t M, size_t N)
{
    // BEGIN YOUR CODE
    // float * A_T = new float[M*K];
    // memset(A_T, 0, M*K*sizeof(float));

    // // #pragma omp parallel for
    // for (size_t i = 0; i < M*K; ++i) {
    //     int a = i/K;
    //     int b = i%K;
    //     A_T[i] = A[M*b+a];
    // }

    // matrix_dot(A_T, B, C, M, K, N);

    memset(C, 0, M*N*sizeof(float));
    
        
    // for (size_t i = 0; i < M; ++i) {
    //     float * sum = &C[i*N];
    //     for (size_t k = 0; k < K; ++k) {
    //         float A1 = A[k*M+i];
    //         const float * B_ptr = &B[k*N];
    //         for (size_t j = 0; j < N; ++j) {
    //             sum[j] +=  B_ptr[j] *A1;
    //         }
    //     }
    // }

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                C[i*N+j] +=  B[k*N+j]* A[k*M+i] ;
            }
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size M * K
 *     B (const float*): Matrix of size N * K
 *     C (float*): Matrix of size M * N
 **/
void matrix_trans_dot(const float *A, const float *B, float *C, size_t M, size_t K, size_t N)
{
    // BEGIN YOUR CODE
    float * B_T = new float[N*K]; //K*N
    // #pragma omp parallel for
    for (size_t i = 0; i < N*K; ++i) {
        int a = i/N;
        int b = i%N;
        B_T[i] = B[K*b+a];
    }

    
    matrix_dot(A, B_T, C, M, K, N);
    // END YOUR CODE
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t M, size_t N)
{
    // BEGIN YOUR CODE
    // float * A_p = &A[0];
    // const float * B_p = &B[0];

    for (size_t i = 0; i < M*N; i++){
        A[i]-=B[i];
        // *A_p -= *B_p;
        // A_p++;
        // B_p++;
    }
    // END YOUR CODE
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t M, size_t N)
{
    // BEGIN YOUR CODE
    // float * C_p = &C[0];
    // for (size_t i = 0; i < M*N; i++){
    //     // A[i]-=B[i];
    //     *C_p *= scalar;
    //     C_p++;
    // }

    for (size_t i = 0; i < M*N; i++){
        C[i] *= scalar;
    }
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t M, size_t N)
{
    for (size_t i = 0; i < M*N; i++){
        C[i] /= scalar;
    }
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t M, size_t N)
{
    // BEGIN YOUR CODE
    for (size_t i = 0; i < M; ++i) {
        // Get the current row
        float* row = C + i * N;

        //avoid numerours overflow issues
        // float max = *std::max_element(row, row + N);
        float max = row[0];

        // Compute exponentials
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            row[i] = std::exp(row[i] - max);
            sum += row[i];
        }

        // Normalize 
        for (size_t i = 0; i < N; ++i) {
            row[i] /= sum;
        }
    }
    // END YOUR CODE
}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m, size_t k)
{
    // BEGIN YOUR CODE
    memset(Y, 0, sizeof(float) * m* k);
    for (size_t i = 0; i < m; ++i) {
        size_t label = static_cast<size_t>(y[i]);

        // if (label < k) {
            Y[i * k + label] = 1.0f;
        // } else {
        //     std::cerr << "Error: Label out of range." << std::endl;
        // }
    }
    // END YOUR CODE
}

/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the logits and gradients.
 *
 * Args:
 *     X (const float *): pointer to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): pointer to y data, of size m
 *     theta (float *): pointer to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (int): SGD minibatch size
 *
 * Returns:
 *     (None)
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE
    float *logits = new float[m * k];
    memset(logits, 0, m*k*sizeof(float));

    float *gradients = new float[n * k];
    memset(gradients, 0, n*k*sizeof(float));

    float* Y = new float[m * k];
    for (size_t start = 0; start < m; start += batch){
        size_t length = std::min(start + batch, m) -start;
        matrix_dot(X + start * n, theta, logits, length, n, k);
        matrix_softmax_normalize(logits, length, k);
        vector_to_one_hot_matrix(y + start, Y, length, k);
        matrix_minus(logits, Y, length, k);
        matrix_dot_trans(X + start * n, logits, gradients, length, n, k);
        matrix_mul_scalar(gradients, lr / static_cast<float>(length), n, k);
        matrix_minus(theta, gradients, n, k);
    }
    // Deallocate memory
    delete[] logits;
    delete[] gradients;
    delete[] Y;
    // END YOUR CODE
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    // std::cout<<"f";
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    
    float *train_result = new float[train_data->images_num * num_classes];
    memset(train_result, 0, train_data->images_num * num_classes * sizeof(float));
    float *test_result = new float[test_data->images_num * num_classes];
    memset(test_result, 0, test_data->images_num * num_classes * sizeof(float));
    
    float train_loss, train_err, test_loss, test_err;
    // std::cout<<"test1";
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    // std::cout<<"test2";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE

        softmax_regression_epoch_cpp(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch);

        matrix_dot(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
        matrix_dot(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);

        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        // print_matrix(theta,  train_data->input_dim, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float loss = 0.0f;

    // Allocate memory for one-hot encoding matrix
    float *one_hot_matrix = new float[images_num * num_classes]();
    vector_to_one_hot_matrix(labels_array, one_hot_matrix, images_num, num_classes);

    for (size_t i = 0; i < images_num; ++i)
    {
        // Extract the logits for the current example
        const float *logits = &result[i * num_classes];

        // Compute the softmax for numerical stability
        matrix_softmax_normalize(const_cast<float *>(logits), 1, num_classes);

        // Compute the cross-entropy loss
        for (size_t j = 0; j < num_classes; ++j)
        {
            loss -= one_hot_matrix[i * num_classes + j] * std::log(logits[j]);
        }
    }

    // Free memory for one-hot encoding matrix
    delete[] one_hot_matrix;

    return loss / images_num;
        // END YOUR CODE
}

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    size_t error_count = 0;

    for (size_t i = 0; i < images_num; ++i)
    {
        // Extract the logits for the current example
        const float *logits = &result[i * num_classes];

        // Compute the softmax for numerical stability
        matrix_softmax_normalize(const_cast<float *>(logits), 1, num_classes);

        // Find the predicted class (index with the highest probability)
        size_t predicted_class = std::distance(logits, std::max_element(logits, logits + num_classes));

        // Check if the predicted class matches the true label
        if (predicted_class != static_cast<size_t>(labels_array[i]))
        {
            ++error_count;
        }
    }

    return static_cast<float>(error_count) / images_num;
    // END YOUR CODE
}

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    logits = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD minibatch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

/**
 *Example function to fully train a nn classifier
 **/
void train_nn(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE

        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
