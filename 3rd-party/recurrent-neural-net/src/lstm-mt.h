//
//  lstm-mt.h
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/27/22.
//

#ifndef lstm_mt_h
#define lstm_mt_h

#include "lstm.h"

typedef struct net_cpu_performance
{
    double cpu_time_forward;
    double cpu_time_back;
    double cpu_time_adam;
    
    double time_forward;
    double time_back;
    double time_adam;
} net_cpu_performance_t;

const net_cpu_performance_t * get_cpu_performance(void);

void lstm_backward_propagate(int layers,
                             lstm_model_t** model_layers,
                             lstm_values_cache_t ***cache_layers,
                             int * Y_train,
                             lstm_values_next_cache_t **d_next_layers,
                             lstm_model_t** gradients,
                             int e1, int e3,
                             int use_thread);

void lstm_forward_propagate(int layers,
                            numeric_t *first_layer_input,
                            lstm_model_t** model_layers,
                            lstm_values_cache_t ***caches_layer,
                            int e1, int e2,
                            int use_thread);

void adam_optimize(int layers,
                   lstm_model_t** model_layers,
                   lstm_model_t** gradient_layers,
                   lstm_model_t** M_layers,
                   lstm_model_t** R_layers,
                   unsigned int n,
                   int use_thread);

#endif /* lstm_mt_h */
