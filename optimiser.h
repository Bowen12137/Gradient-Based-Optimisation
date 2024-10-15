#ifndef OPTIMISER_H
#define OPTIMISER_H

#include <stdio.h>

void initialise_optimiser(double learning_rate, int batch_size, int total_epochs, double alpha, double initial_learning_rate, double final_learning_rate);
void run_optimisation(void);
double evaluate_objective_function(unsigned int sample);
void numerical_derivative();

#endif /* OPTMISER_H */
