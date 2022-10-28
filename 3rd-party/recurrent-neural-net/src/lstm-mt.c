//
//  lstm-mt.c
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/27/22.
//

#include "lstm-mt.h"
#include <pthread.h>
#include <time.h>

// thread argument structure definition

// backward
typedef struct back_thread_arg
{
    lstm_model_t* model;
    numeric_t* y_probabilities;
    int y_correct;
    lstm_values_next_cache_t* d_next;
    lstm_values_cache_t* cache_in;
    lstm_model_t* gradients;
    lstm_values_next_cache_t* cache_out;
} back_thread_arg_t;

// forward
typedef struct forward_thread_arg
{
    lstm_model_t* model;
    numeric_t* input;
    lstm_values_cache_t* cache_in;
    lstm_values_cache_t* cache_out;
    int softmax;
} forward_thread_arg_t;

// adam optimizer
typedef struct thread_arg {
    lstm_model_t* model;
    lstm_model_t* gradients;
    lstm_model_t* M;
    lstm_model_t* R;
    unsigned int t;
} thread_arg_t;

typedef struct adam_thread_arg {
    lstm_model_t * _model;
    lstm_model_t * _gradients;
    lstm_model_t * _m_model;
    lstm_model_t * _r_model;
    
    numeric_t beta1;
    numeric_t beta2;
    
    numeric_t beta1t;
    numeric_t beta2t;
    
    int size1;
    int size2;
    int index;
    unsigned int t;
} adam_thread_arg_t;

static net_cpu_performance_t performance = {0};

static double timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
    int64_t diff = ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
                   ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
    return (double)diff/1000000000.0;
}

const net_cpu_performance_t * get_cpu_performance(void)
{
    return (const net_cpu_performance_t *)&performance;
}

static void * lstm_back_thread(void * vargp)
{
    back_thread_arg_t * arg = (back_thread_arg_t *)vargp;
    lstm_backward_propagate_layer(arg->model,
                                     arg->y_probabilities,                // d_next_layers[p-1]->dldY_pass,
                                     arg->y_correct,                      // -1
                                     arg->d_next,
                                     arg->cache_in,
                                     arg->gradients,
                                     arg->cache_out);
    return NULL;
}


// model, input, state and cache values, &probs, whether or not to apply softmax
void lstm_backward_propagate(int layers,
                             lstm_model_t** model_layers,
                             lstm_values_cache_t ***cache_layers,
                             int * Y_train,
                             lstm_values_next_cache_t **d_next_layers,
                             lstm_model_t** gradients,
                             int e1,
                             int e3,
                             int use_thread)
{
    time_t prg_begin, prg_end;
    int p;
    struct timespec start, end;
      
    clock_gettime(CLOCK_MONOTONIC, &start);
    prg_begin = clock();
    if (use_thread && layers > 1)
    {
        const int size = layers-1;
        pthread_t thread_ids[size];
        back_thread_arg_t args[size];
                
        for ( p = 0; p < size; p++ )
        {
            args[p].y_probabilities = (p==0) ? cache_layers[p][e1]->probs : d_next_layers[p-1]->dldY_pass;
            args[p].y_correct = (p==0) ? Y_train[e3] : -1;
            args[p].model = model_layers[p];
            args[p].gradients = gradients[p];
            args[p].d_next = d_next_layers[p];
            args[p].cache_in = cache_layers[p][e1];
            args[p].cache_out = d_next_layers[p];
            pthread_create(&thread_ids[p], NULL, lstm_back_thread, &args[p]);
        }
        // run this on main thread
        lstm_backward_propagate_layer(model_layers[p],
                                         d_next_layers[p-1]->dldY_pass,
                                         -1,
                                         d_next_layers[p],
                                         cache_layers[p][e1],
                                         gradients[p],
                                         d_next_layers[p]);
        for (p = 0; p < size; p++)
        {
            pthread_join(thread_ids[p], NULL);
        }
    }
    else
    {
        for (p = 0; p < layers; p++)
        {
            numeric_t * y_probabilities = (p==0) ? cache_layers[p][e1]->probs : d_next_layers[p-1]->dldY_pass;
            int y_correct = (p==0) ? Y_train[e3] : -1;
            lstm_backward_propagate_layer(model_layers[p],
                                             y_probabilities,                // d_next_layers[p-1]->dldY_pass,
                                             y_correct,                      // -1
                                             d_next_layers[p],
                                             cache_layers[p][e1],
                                             gradients[p],
                                             d_next_layers[p]);

        }
    }
    prg_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &end);
    performance.cpu_time_back += (double)(prg_end - prg_begin) / (double)CLOCKS_PER_SEC;
    performance.time_back += timespecDiff(&end, &start);
}


static void * lstm_forward_thread(void * vargp)
{
    forward_thread_arg_t * arg = (forward_thread_arg_t *)vargp;
    lstm_forward_propagate_layer(arg->model,
                                 arg->input,
                                 arg->cache_in,
                                 arg->cache_out,
                                 arg->softmax);
    return NULL;
}


// model, input, state and cache values, &probs, whether or not to apply softmax
void lstm_forward_propagate(int layers,
                            numeric_t *first_layer_input,
                            lstm_model_t** model_layers,
                            lstm_values_cache_t ***caches_layer,
                            int e1, int e2,
                            int use_thread)
{
    time_t prg_begin, prg_end;
    int p;
    struct timespec start, end;
      
    clock_gettime(CLOCK_MONOTONIC, &start);
    prg_begin = clock();
    if (use_thread && layers > 1)
    {
        pthread_t thread_ids[layers];
        forward_thread_arg_t args[layers];
        
        for (p = layers-1; p >= 0; p--)
        {
            args[p].input = (p==layers-1) ? first_layer_input : caches_layer[p+1][e2]->probs;
            args[p].model = model_layers[p];
            args[p].cache_in = caches_layer[p][e1];
            args[p].cache_out = caches_layer[p][e2];
            args[p].softmax = (p == 0);
            pthread_create(&thread_ids[p], NULL, lstm_forward_thread, &args[p]);
        }
        for (p = 0; p < layers; p++)
        {
            pthread_join(thread_ids[p], NULL);
        }
    }
    else
    {
        for (p = layers-1; p >= 0; p--)
        {
            numeric_t * input = (p==layers-1) ? first_layer_input : caches_layer[p+1][e2]->probs;
            lstm_forward_propagate_layer(model_layers[p],
                                         input,
                                         caches_layer[p][e1],
                                         caches_layer[p][e2],
                                         p==0);
        }
    }
    prg_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &end);
    performance.cpu_time_forward += (double)(prg_end - prg_begin) / (double)CLOCKS_PER_SEC;
    performance.time_forward += timespecDiff(&end, &start);
}

static void * optimize_thread(void * vargp)
{
    thread_arg_t * arg = (thread_arg_t *)vargp;
    gradients_adam_optimizer(
                             arg->model,
                             arg->gradients,
                             arg->M,
                             arg->R,
                             arg->t);
    return NULL;
}

static void * optimize_weight_thread(void * vargp)
{
    adam_thread_arg_t * arg = (adam_thread_arg_t *)vargp;
    int i = arg->index;

    // will overwrite if file exists
    vectors_copy_multiply_scalar(arg->_gradients->Wm[i], arg->_gradients->W[i], 1.0 - arg->beta1, arg->size1);
    vectors_copy_multiply_scalar(arg->_gradients->bm[i], arg->_gradients->b[i], 1.0 - arg->beta1, arg->size2);
    vectors_multiply_scalar(arg->_m_model->W[i], arg->beta1, arg->size1);
    vectors_multiply_scalar(arg->_m_model->b[i], arg->beta1, arg->size2);
    vectors_add(arg->_m_model->W[i], arg->_gradients->W[i], arg->size1);
    vectors_add(arg->_m_model->b[i], arg->_gradients->b[i], arg->size2);
    // M Done!
    
    // Computing R
    vectors_multiply(arg->_gradients->W[i], arg->_gradients->W[i], arg->size1);
    vectors_multiply(arg->_gradients->b[i], arg->_gradients->b[i], arg->size2 );
    vectors_copy_multiply_scalar(arg->_gradients->Wm[i], arg->_gradients->W[i], 1.0 - arg->beta2, arg->size1);
    vectors_copy_multiply_scalar(arg->_gradients->bm[i], arg->_gradients->b[i], 1.0 - arg->beta2, arg->size2);
    vectors_multiply_scalar(arg->_r_model->W[i], arg->beta2, arg->size1);
    vectors_multiply_scalar(arg->_r_model->b[i], arg->beta2, arg->size2);
    vectors_add(arg->_r_model->W[i], arg->_gradients->W[i], arg->size1);
    vectors_add(arg->_r_model->b[i], arg->_gradients->b[i], arg->size2);
    // R done!
    
    vectors_copy_multiply_scalar(arg->_m_model->Wm[i], arg->_m_model->W[i], arg->beta1t, arg->size1);
    vectors_copy_multiply_scalar(arg->_m_model->bm[i], arg->_m_model->b[i], arg->beta1t, arg->size2);
    // M hat done!
    
    vectors_copy_multiply_scalar(arg->_r_model->Wm[i], arg->_r_model->W[i], arg->beta2t, arg->size1);
    vectors_copy_multiply_scalar(arg->_r_model->bm[i], arg->_r_model->b[i], arg->beta2t, arg->size2);
    // R hat done!
    
    vector_sqrt(arg->_r_model->Wm[i], arg->size1);
    vector_sqrt(arg->_r_model->bm[i], arg->size2);
    vectors_add_scalar(arg->_r_model->Wm[i], EPSILON, arg->size1);
    vectors_add_scalar(arg->_r_model->bm[i], EPSILON, arg->size2);
    vectors_copy_multiply_scalar(arg->_gradients->Wm[i], arg->_m_model->Wm[i], arg->_model->params->learning_rate, arg->size1);
    vectors_copy_multiply_scalar(arg->_gradients->bm[i], arg->_m_model->bm[i], arg->_model->params->learning_rate, arg->size2);
    vectors_div(arg->_gradients->Wm[i], arg->_r_model->Wm[i], arg->size1);
    vectors_div(arg->_gradients->bm[i], arg->_r_model->bm[i], arg->size2);
    vectors_subtract(arg->_model->W[i], arg->_gradients->Wm[i], arg->size1);
    vectors_subtract(arg->_model->b[i], arg->_gradients->bm[i], arg->size2);

    return NULL;
}


static void adam_optimize_layer(lstm_model_t* model,
                                lstm_model_t* gradients,
                                lstm_model_t* M,
                                lstm_model_t* R,
                                unsigned int t,
                                int use_thread)
{
    if (use_thread)
    {
        pthread_t thread_ids[LSTM_PARAMTERS];
        adam_thread_arg_t args[LSTM_PARAMTERS];
        int i;
        
        for (i = 0; i < LSTM_PARAMTERS; i++)
        {
            args[i].beta1 = model->params->beta1;
            args[i].beta2 = model->params->beta2;
            args[i].size1 = (i == LSTM_WB_Y) ? model->Y * model->N : model->N * model->S;
            args[i].size2 = (i == LSTM_WB_Y) ? model->Y : model->N;
            args[i].beta1t = 1.0 / ( 1.0 - pow(args[i].beta1, t+1));
            args[i].beta2t = 1.0 / ( 1.0 - pow(args[i].beta2, t+1));
            args[i].index = i;
            args[i]._gradients = gradients;
            args[i]._model = model;
            args[i]._m_model = M;
            args[i]._r_model = R;
            args[i].t = t;
            pthread_create(&thread_ids[i], NULL, optimize_weight_thread, &args[i]);
        }
        
        for (i = 0; i < LSTM_PARAMTERS; i++)
        {
            pthread_join(thread_ids[i], NULL);
        }
    }
    else
    {
        gradients_adam_optimizer(model, gradients, M, R, t);
    }
}

void adam_optimize(int layers,
                   lstm_model_t** model_layers,
                   lstm_model_t** gradient_layers,
                   lstm_model_t** M_layers,
                   lstm_model_t** R_layers,
                   unsigned int n,
                   int use_thread)
{
    time_t  prg_begin, prg_end;
    int     p;
    struct timespec start, end;
      
    clock_gettime(CLOCK_MONOTONIC, &start);
    prg_begin = clock();
    if (use_thread && layers > 1)
    {
        const int size = layers-1;
        pthread_t thread_ids[size];
        thread_arg_t args[size];

        for ( p = 0; p < size; p++ )
        {
            args[p].model = model_layers[p];
            args[p].gradients = gradient_layers[p];
            args[p].R = R_layers[p];
            args[p].M = M_layers[p];
            args[p].t = n;
            pthread_create(&thread_ids[p], NULL, optimize_thread, &args[p]);
        }
        adam_optimize_layer(model_layers[p],
                            gradient_layers[p],
                            M_layers[p],
                            R_layers[p],
                            n,
                            use_thread);
        for (p = 0; p < size; p++)
        {
            pthread_join(thread_ids[p], NULL);
        }
    }
    else
    {
        for ( p = 0; p < layers; p++ )
        {
            adam_optimize_layer(model_layers[p],
                                gradient_layers[p],
                                M_layers[p],
                                R_layers[p],
                                n,
                                use_thread);
        }
    }
    prg_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &end);
    performance.cpu_time_adam += (double)(prg_end - prg_begin) / (double)CLOCKS_PER_SEC;
    performance.time_adam += timespecDiff(&end, &start);
}
