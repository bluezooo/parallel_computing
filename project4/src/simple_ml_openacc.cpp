#include "simple_ml_openacc.hpp"

void matrix_dot_openacc(const float *A, const float *B,
                        float *C, size_t M, size_t K, size_t N)
{

    memset(C, 0, M*N*sizeof(float));

    #pragma acc enter data copyin(A[0 : M*K], buffer[0 : width * height * num_channels],\
                    filter[0: FILTER_SIZE], width, height, num_channels, FILTER_SIZE)
    #pragma acc update device(filteredImage[0 : width * height* num_channels], \
                    buffer[0 : width * height * num_channels],\
                    filter[0: FILTER_SIZE], width, height, num_channels, FILTER_SIZE)
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                C[i*N+j] += B[k*N+j] * A[i*K+k];
            }
        }
    }
}

void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t K, size_t M, size_t N)
{
    // BEGIN YOUR CODE


    memset(C, 0, M*N*sizeof(float));
    

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                C[i*N+j] +=  B[k*N+j]* A[k*M+i] ;
            }
        }
    }
}

void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_minus_openacc(float *A, const float *B, size_t M, size_t N)
{
    // BEGIN YOUR CODE
    for (size_t i = 0; i < M*N; i++){
        A[i]-=B[i];
    }
    // END YOUR CODE
}

void matrix_mul_scalar_openacc(float *C, float scalar, size_t M, size_t N)
{
    // BEGIN YOUR CODE
    for (size_t i = 0; i < M*N; i++){
        C[i] *= scalar;
    }
    // END YOUR CODE
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t M, size_t N)
{
    // BEGIN YOUR CODE
    for (size_t i = 0; i < M*N; i++){
        C[i] /= scalar;
    }
    // END YOUR CODE
}

void matrix_softmax_normalize_openacc(float *C, size_t M, size_t N)
{
    // BEGIN YOUR CODE
    for (size_t i = 0; i < M; ++i) {
        float* row = C + i * N;

        // float max = *std::max_element(row, row + N);
        float max = row[0];
        
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            row[i] = std::exp(row[i] - max);
            sum += row[i];
        }

        for (size_t i = 0; i < N; ++i) {
            row[i] /= sum;
        }
    }
    // END YOUR CODE
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t k)
{
    // BEGIN YOUR CODE
    for (size_t i = 0; i < m; ++i) {
        size_t label = static_cast<size_t>(y[i]);

        Y[i * k + label] = 1.0f;

    }
    // END YOUR CODE
}

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch)
{
    // BEGIN YOUR CODE
    float *logits = new float[m * k];
    memset(logits, 0, m*k*sizeof(float));

    float *gradients = new float[n * k];
    memset(gradients, 0, n*k*sizeof(float));

    float* Y = new float[m * k];
    for (size_t start = 0; start < m; start += batch){
        size_t length = std::min(start + batch, m) -start;
        matrix_dot_openacc(X + start * n, theta, logits, length, n, k);
        matrix_softmax_normalize_openacc(logits, length, k);
        vector_to_one_hot_matrix_openacc(y + start, Y, length, k);
        matrix_minus_openacc(logits, Y, length, k);
        matrix_dot_trans_openacc(X + start * n, logits, gradients, length, n, k);
        matrix_mul_scalar_openacc(gradients, lr / static_cast<float>(length), n, k);
        matrix_minus_openacc(theta, gradients, n, k);
    }
    // Deallocate memory
    delete[] logits;
    delete[] gradients;
    delete[] Y;
    // END YOUR CODE
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    /*
    Example function to fully train a softmax regression classifier
    */
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    
    float *train_result = new float[train_data->images_num * num_classes];
    memset(train_result, 0, train_data->images_num * num_classes * sizeof(float));
    float *test_result = new float[test_data->images_num * num_classes];
    memset(test_result, 0, test_data->images_num * num_classes * sizeof(float));
    
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        softmax_regression_epoch_cpp_openacc(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch);

        matrix_dot_openacc(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
        matrix_dot_openacc(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);
        
        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
   // END YOUR CODE
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float loss = 0.0f;
    float *one_hot_matrix = new float[images_num * num_classes]();
    vector_to_one_hot_matrix_openacc(labels_array, one_hot_matrix, images_num, num_classes);

    for (size_t i = 0; i < images_num; ++i)
    {
        const float *logits = &result[i * num_classes];
        matrix_softmax_normalize_openacc(const_cast<float *>(logits), 1, num_classes);

        // cross-entropy loss
        for (size_t j = 0; j < num_classes; ++j)
        {
            loss -= one_hot_matrix[i * num_classes + j] * std::log(logits[j]);
        }
    }
    delete[] one_hot_matrix;
    return loss / images_num;
    // END YOUR CODE
}

float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    size_t error_count = 0;

    for (size_t i = 0; i < images_num; ++i)
    {
        // Extract the logits for the current example
        const float *logits = &result[i * num_classes];

        // Compute the softmax for numerical stability
        matrix_softmax_normalize_openacc(const_cast<float *>(logits), 1, num_classes);

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

void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
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
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
