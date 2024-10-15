#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"

// Function declarations
void update_parameters(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

// Paramters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double learning_rate;
double i_lr;
double f_lr;
int N_NEURONS;
double alpha;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy){
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter, total_iter, mean_loss, test_accuracy);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs, double cmd_alpha, double cmd_initial_learning_rate, double cmd_final_learning_rate){
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;

    alpha = cmd_alpha;
    i_lr = cmd_initial_learning_rate;
    f_lr = cmd_final_learning_rate;
    
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with paramters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n",
           total_epochs, batch_size, num_batches, learning_rate);
}

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    // Open file for writing objective function
    FILE *obj_file = fopen("objective_function_results.txt", "w");
    if (obj_file == NULL) {
        printf("Error opening obj_file for writing\n");
        return;
    }
    
    // Open file for writing test accuracy
    FILE *acc_file = fopen("test_accuracy_results.txt", "w");
    if (acc_file == NULL) {
        printf("Error opening acc_file for writing\n");
        return;
    }
    // Run optimiser - update parameters after each minibatch
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){
            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            
            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0){
                printf("Epoch: %u,  Total iter: %u,  Iter Loss: %0.12f, ", epoch_counter, total_iter, obj_func);
                test_accuracy = evaluate_testing_accuracy();
                printf("Test Acc: %f\n", test_accuracy);
                fprintf(obj_file, "%f\n", obj_func);
                fprintf(acc_file, "%f\n", test_accuracy);
            }
            
            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;
            }

        }
            update_parameters(batch_size);
            // learning_rate_decay(epoch_counter);
            // batch_size_increase(epoch_counter);
            // update_parameters(batch_size);
            // learning_rate_decay(epoch_counter);
            // Update weights on batch completion
            // update_parameters_momentum(batch_size, epoch_counter);
            // update_parameters_momentum(batch_size, epoch_counter);
            // update_parameters_adam(batch_size);
        
    }
    
    // Print final performance
    printf("Epoch: %u,  Total iter: %u,  Iter Loss: %0.12f, ", epoch_counter, total_iter, obj_func);
    test_accuracy = evaluate_testing_accuracy();
    printf("Test Acc: %f\n\n", test_accuracy);
    fprintf(obj_file, "%f\n", obj_func);
    fprintf(acc_file, "%f\n", test_accuracy);
}

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    
    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    
    // Evaluate parameter updates
    store_gradient_contributions();
    
    return loss;
}

void numerical_derivative(unsigned int sample){
    // w_LI_L1[0][0].dw = 0;
    // double first = evaluate_objective_function(sample);
    // w_LI_L1[0][0].w = w_LI_L1[0][0].w + 0.05;
    // w_LI_L1[0][0].dw = 0;
    // double second = evaluate_objective_function(sample+1);
    // double derivative = (learning_rate/batch_size) * ((second - first)/0.15);
    // double analytic = w_LI_L1[0][0].dw;
    // double analytic2 = dL_dW_LI_L1[0][0];
    // printf("detivative= %f, Analytical Solution = %f, Analytical Solution 2 = %f\n", derivative, analytic, analytic2);


    w_L1_L2[0][0].dw = 0;
    double old_loss = evaluate_objective_function(sample);
    w_L1_L2[0][0].w = w_L1_L2[0][0].w + 0.05;
    w_L1_L2[0][0].dw = 0;
    double new_loss = evaluate_objective_function(sample+1);
    double detivative = (learning_rate/batch_size) * ((new_loss - old_loss)/0.05);
    double analytic = w_L1_L2[0][0].dw;
    double analytic2 = dL_dW_L1_L2[0][0];
    printf(" detivative = %f, Analytical Solution = %f, Analytical Solution 2 = %f\n", detivative, analytic, analytic2);
}

void update_weights(weight_struct_t weights[][N_NEURONS], int n_rows, int n_cols, unsigned int batch_size) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            weights[i][j].w -= ((learning_rate / batch_size) * weights[i][j].dw);
            weights[i][j].dw = 0.0;
        }
    }
}

void update_parameters(unsigned int batch_size) {
    // Update weights for each layer
    update_weights(w_LI_L1, N_NEURONS_LI, N_NEURONS_L1, batch_size);
    update_weights(w_L1_L2, N_NEURONS_L1, N_NEURONS_L2, batch_size);
    update_weights(w_L2_L3, N_NEURONS_L2, N_NEURONS_L3, batch_size);
    update_weights(w_L3_LO, N_NEURONS_L3, N_NEURONS_LO, batch_size);
}


void learning_rate_decay(unsigned int epoch_counter){
    learning_rate = i_lr * (1 - (epoch_counter/total_epochs)) + ((epoch_counter/total_epochs) * f_lr);
}


void update_weights_momentum(weight_struct_t weights[][N_NEURONS], int n_rows, int n_cols, unsigned int batch_size, double alpha, double learning_rate) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            weights[i][j].v = (alpha * weights[i][j].v) - ((learning_rate / batch_size) * weights[i][j].dw);
            weights[i][j].w += weights[i][j].v;
            weights[i][j].dw = 0.0;
        }
    }
}

void update_parameters_momentum(unsigned int batch_size, unsigned int epoch_counter) {
    // Update weights for each layer
    update_weights_momentum(w_LI_L1, N_NEURONS_LI, N_NEURONS_L1, batch_size, alpha, learning_rate);
    update_weights_momentum(w_L1_L2, N_NEURONS_L1, N_NEURONS_L2, batch_size, alpha, learning_rate);
    update_weights_momentum(w_L2_L3, N_NEURONS_L2, N_NEURONS_L3, batch_size, alpha, learning_rate);
    update_weights_momentum(w_L3_LO, N_NEURONS_L3, N_NEURONS_LO, batch_size, alpha, learning_rate);
}


void update_weights_adam(weight_struct_t weights[][N_NEURONS], int n_rows, int n_cols, unsigned int batch_size, double beta_1, double beta_2, double epsilon, double learning_rate) {
    double bias_corrected_mean;
    double bias_corrected_variance;

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            weights[i][j].mean = (beta_1 * weights[i][j].mean) + ((1 - beta_1) * weights[i][j].dw);
            weights[i][j].variance = (beta_2 * weights[i][j].variance) + ((1 - beta_2) * (weights[i][j].dw * weights[i][j].dw));
            
            bias_corrected_mean = (weights[i][j].mean / (1 - beta_1));
            bias_corrected_variance = (weights[i][j].variance / (1 - beta_2));
            
            weights[i][j].w -= ((learning_rate / ((batch_size * sqrt(bias_corrected_variance)) + epsilon)) * bias_corrected_mean);
            weights[i][j].dw = 0.0;
        }
    }
}

void update_parameters_adam(unsigned int batch_size) {
    double beta_1 = 0.9;
    double beta_2 = 0.9999;
    double epsilon = 0.00000001;

    // Update weights for each layer
    update_weights_adam(w_LI_L1, N_NEURONS_LI, N_NEURONS_L1, batch_size, beta_1, beta_2, epsilon, learning_rate);
    update_weights_adam(w_L1_L2, N_NEURONS_L1, N_NEURONS_L2, batch_size, beta_1, beta_2, epsilon, learning_rate);
    update_weights_adam(w_L2_L3, N_NEURONS_L2, N_NEURONS_L3, batch_size, beta_1, beta_2, epsilon, learning_rate);
    update_weights_adam(w_L3_LO, N_NEURONS_L3, N_NEURONS_LO, batch_size, beta_1, beta_2, epsilon, learning_rate);
}
