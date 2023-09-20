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

// adam optimizer
typedef struct thread_arg {
    lstm_model_t* model;
    lstm_model_t* gradients;
    lstm_model_t* M;
    lstm_model_t* R;
    unsigned int t;
    int use_thread;
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

typedef struct back_args
{
    lstm_values_cache_t * cache_in;
    lstm_model_t * model;
    lstm_model_t * gradients;
    numeric_t *    dldh;
    numeric_t *    dldc;
    int N;
    int S;
} back_args_t;

static void * back_thread_hf(void * args)
// can split in 4 threads
{
    back_args_t * a = (back_args_t *)args;
    numeric_t * dldhf = a->model->dldhf;

    vectors_copy_multiply(dldhf, a->dldc, a->cache_in->c_old, a->N);
    sigmoid_backward(dldhf, a->cache_in->hf, dldhf, a->N);
    fully_connected_backward(dldhf, a->model->W[LSTM_WB_F], a->cache_in->X, a->gradients->W[LSTM_WB_F], a->gradients->dldXf, a->gradients->b[LSTM_WB_F], a->N, a->S);

    return NULL;
}

static void * back_thread_hi(void * args)
{
    back_args_t * a = (back_args_t *)args;
    numeric_t * dldhi = a->model->dldhi;
    
    vectors_copy_multiply(dldhi, a->cache_in->hc, a->dldc, a->N);
    sigmoid_backward(dldhi, a->cache_in->hi, dldhi, a->N);
    fully_connected_backward(dldhi, a->model->W[LSTM_WB_I], a->cache_in->X, a->gradients->W[LSTM_WB_I], a->gradients->dldXi, a->gradients->b[LSTM_WB_I], a->N, a->S);

    return NULL;
}

static void * back_thread_ho(void * args)
{
    back_args_t * a = (back_args_t *)args;
    numeric_t * dldho = a->model->dldho;

    vectors_copy_multiply(dldho, a->dldh, a->cache_in->tanh_c_cache, a->N);
    sigmoid_backward(dldho, a->cache_in->ho, dldho, a->N);
    fully_connected_backward(dldho, a->model->W[LSTM_WB_O], a->cache_in->X, a->gradients->W[LSTM_WB_O], a->gradients->dldXo, a->gradients->b[LSTM_WB_O], a->N, a->S);

    return NULL;
}

static void * back_thread_hc(void * args)
{
    back_args_t * a = (back_args_t *)args;
    numeric_t * dldhc = a->model->dldhc;

    vectors_copy_multiply(dldhc, a->cache_in->hi, a->dldc, a->N);
    
    if (a->model->params->use_tanf)
        tanf_backward(dldhc, a->cache_in->hc, dldhc, a->N);
    else
        tanh_backward(dldhc, a->cache_in->hc, dldhc, a->N);
    fully_connected_backward(dldhc, a->model->W[LSTM_WB_C], a->cache_in->X, a->gradients->W[LSTM_WB_C], a->gradients->dldXc, a->gradients->b[LSTM_WB_C], a->N, a->S);
    return NULL;
}


// model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
static void lstm_backward_propagate_layer(lstm_model_t* model, numeric_t* y_probabilities, int y_correct,
                             lstm_values_next_cache_t* d_next, lstm_values_cache_t* cache_in,
                             lstm_model_t* gradients, lstm_values_next_cache_t* cache_out)
{
    numeric_t *h,*dldh_next,*dldc_next, *dldy, *dldh, *dldc;
    int i, N, Y, S;
    
    N = model->N;
    Y = model->Y;
    S = model->S;
    
    // model cache
    dldh = model->dldh;
    dldc = model->dldc;
    
    h = cache_in->h;
    
    dldh_next = d_next->dldh_next;
    dldc_next = d_next->dldc_next;
    
    dldy = y_probabilities;
    
    if ( y_correct >= 0 )
    {
        dldy[y_correct] -= 1.0;
    }
#ifdef INTERLAYER_SIGMOID_ACTIVATION
    if ( y_correct < 0 )
    {
        sigmoid_backward(dldy, cache_in->probs_before_sigma, dldy, Y);
    }
#endif
    
    fully_connected_backward(dldy, model->W[LSTM_WB_Y], h, gradients->W[LSTM_WB_Y], dldh, gradients->b[LSTM_WB_Y], Y, N);
    vectors_add(dldh, dldh_next, N);
        
    vectors_copy_multiply(dldc, dldh, cache_in->ho, N);
    
    if (model->params->use_tanf)
        tanf_backward(dldc, cache_in->tanh_c_cache, dldc, N);
    else
        tanh_backward(dldc, cache_in->tanh_c_cache, dldc, N);

    vectors_add(dldc, dldc_next, N);
        
    // TODO: 4 threads can be created to speed up

    if (model->params->use_threads)
    {
        const int size = LSTM_PARAMTERS-1;
        pthread_t thread_ids[size];
        back_args_t args;
        
        args.model = model;
        args.N = N;
        args.S = S;
        args.cache_in = cache_in;
        args.gradients = gradients;
        args.dldc = dldc;
        args.dldh = dldh;
        
        for (i = 0; i < size; i++)
        {
            switch(i)
            {
                case LSTM_WB_C :
                    pthread_create(&thread_ids[i], NULL, back_thread_hc, &args);
                    break;
                    
                case LSTM_WB_O :
                    pthread_create(&thread_ids[i], NULL, back_thread_ho, &args);
                    break;
                    
                case LSTM_WB_F :
                    pthread_create(&thread_ids[i], NULL, back_thread_hf, &args);
                    break;
                    
                case LSTM_WB_I :
                    pthread_create(&thread_ids[i], NULL, back_thread_hi, &args);
                    break;
                    
                default:
                    printf("\r\nInternal Error #102, Please report!\r\n");
                    exit(0);
                    break;
            }
        }
        for (i = 0; i < size; i++)
        {
            pthread_join(thread_ids[i], NULL);
        }
    }   
    else 
    {
        numeric_t * dldho = model->dldho;
        numeric_t * dldhc = model->dldhc;
        numeric_t * dldhf = model->dldhf;
        numeric_t * dldhi = model->dldhi;

        // ho
        vectors_copy_multiply(dldho, dldh, cache_in->tanh_c_cache, N);
        sigmoid_backward(dldho, cache_in->ho, dldho, N);
        fully_connected_backward(dldho, model->W[LSTM_WB_O], cache_in->X, gradients->W[LSTM_WB_O], gradients->dldXo, gradients->b[LSTM_WB_O], N, S);
    
        // hc
        vectors_copy_multiply(dldhc, cache_in->hi, dldc, N);
        
        if (model->params->use_tanf)
            tanf_backward(dldhc, cache_in->hc, dldhc, N);
        else
            tanh_backward(dldhc, cache_in->hc, dldhc, N);
        fully_connected_backward(dldhc, model->W[LSTM_WB_C], cache_in->X, gradients->W[LSTM_WB_C], gradients->dldXc, gradients->b[LSTM_WB_C], N, S);

        // hc -> hf
        vectors_copy_multiply(dldhf, dldc, cache_in->c_old, N);
        sigmoid_backward(dldhf, cache_in->hf, dldhf, N);
        fully_connected_backward(dldhf, model->W[LSTM_WB_F], cache_in->X, gradients->W[LSTM_WB_F], gradients->dldXf, gradients->b[LSTM_WB_F], N, S);
    
        // hc -> hi
        vectors_copy_multiply(dldhi, cache_in->hc, dldc, N);
        sigmoid_backward(dldhi, cache_in->hi, dldhi, N);
        fully_connected_backward(dldhi, model->W[LSTM_WB_I], cache_in->X, gradients->W[LSTM_WB_I], gradients->dldXi, gradients->b[LSTM_WB_I], N, S);
    }
    
    // dldXi will work as a temporary substitute for dldX (where we get extract dh_next from!)
    vectors_add(gradients->dldXi, gradients->dldXc, S);
    vectors_add(gradients->dldXi, gradients->dldXo, S);
    vectors_add(gradients->dldXi, gradients->dldXf, S);
    
    copy_vector(cache_out->dldh_next, gradients->dldXi, N);
    vectors_copy_multiply(cache_out->dldc_next, cache_in->hf, dldc, N);
    
    // To pass on to next layer
    copy_vector(cache_out->dldY_pass, &gradients->dldXi[N], model->X);
}


// model, input, state and cache values, &probs, whether or not to apply softmax
void lstm_backward_propagate(int layers,
                             lstm_model_t** model_layers,
                             lstm_values_cache_t ***cache_layers,
                             int * Y_train,
                             lstm_values_next_cache_t **d_next_layers,
                             lstm_model_t** gradients,
                             int e1,
                             int e3)
{
    time_t prg_begin, prg_end;
    int p;
    struct timespec start, end;
      
    clock_gettime(CLOCK_MONOTONIC, &start);
    prg_begin = clock();
    
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

    prg_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &end);
    performance.cpu_time_back += (double)(prg_end - prg_begin) / (double)CLOCKS_PER_SEC;
    performance.time_back += timespecDiff(&end, &start);
}

typedef struct forward_args
{
    lstm_values_cache_t* cache_out;
    lstm_model_t* model;
    numeric_t * X_one_hot;
    int N;
    int S;
} forward_args_t;


static void * forward_thread_hf(void * args)
// can split in 4 threads
{
    forward_args_t * a = (forward_args_t *)args;
    fully_connected_forward(a->cache_out->hf, a->model->W[LSTM_WB_F], a->X_one_hot, a->model->b[LSTM_WB_F], a->N, a->S);
    sigmoid_forward(a->cache_out->hf, a->cache_out->hf, a->N);
    return NULL;
}

static void * forward_thread_hi(void * args)
{
    forward_args_t * a = (forward_args_t *)args;
    fully_connected_forward(a->cache_out->hi, a->model->W[LSTM_WB_I], a->X_one_hot, a->model->b[LSTM_WB_I], a->N, a->S);
    sigmoid_forward(a->cache_out->hi, a->cache_out->hi, a->N);
    return NULL;
}

static void * forward_thread_ho(void * args)
{
    forward_args_t * a = (forward_args_t *)args;

    fully_connected_forward(a->cache_out->ho, a->model->W[LSTM_WB_O], a->X_one_hot, a->model->b[LSTM_WB_O], a->N, a->S);
    sigmoid_forward(a->cache_out->ho, a->cache_out->ho, a->N);
    return NULL;
}

static void * forward_thread_hc(void * args)
{
    forward_args_t * a = (forward_args_t *)args;

    fully_connected_forward(a->cache_out->hc, a->model->W[LSTM_WB_C], a->X_one_hot, a->model->b[LSTM_WB_C], a->N, a->S);
    if (a->model->params->use_tanf)
        tanf_forward(a->cache_out->hc, a->cache_out->hc, a->N);
    else
        tanh_forward(a->cache_out->hc, a->cache_out->hc, a->N);
    return NULL;
}

static void lstm_forward_propagate_layer(lstm_model_t* model, numeric_t *input,
                            lstm_values_cache_t* cache_in, lstm_values_cache_t* cache_out,
                            int softmax)
{
    int N, Y, S, i = 0;
    numeric_t *h_old, *c_old, *X_one_hot;
    
    h_old = cache_in->h;
    c_old = cache_in->c;
    
    N = model->N;
    Y = model->Y;
    S = model->S;
    
#ifdef _WIN32
    // MSVC is not a C99 compiler, and does not support variable length arrays
    // MSVC is documented as conforming to C90
    numeric_t *tmp;
    if ( init_zero_vector(&tmp, N) )
    {
        fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n",
                __FILE__, __func__, __LINE__, N);
        exit(1);
    }
#else
    numeric_t tmp[N]; // VLA must be supported.. May cause portability problems.. If so use init_zero_vector (will be slower).
#endif
    
    copy_vector(cache_out->h_old, h_old, N);
    copy_vector(cache_out->c_old, c_old, N);
    
    X_one_hot = cache_out->X;
    
    while ( i < S )
    {
        if ( i < N )
        {
            X_one_hot[i] = h_old[i];
        }
        else
        {
            X_one_hot[i] = input[i - N];
        }
        ++i;
    }
    
    // Fully connected + sigmoid layers

    // can split in 4 threads
    if (model->params->use_threads)
    {
        const int size = LSTM_PARAMTERS-1;
        pthread_t thread_ids[size];
        forward_args_t args;
        
        args.model = model;
        args.N = N;
        args.S = S;
        args.X_one_hot = X_one_hot;
        args.cache_out = cache_out;
        
        for (i = 0; i < size; i++)
        {
            switch(i)
            {
                case LSTM_WB_C :
                    pthread_create(&thread_ids[i], NULL, forward_thread_hc, &args);
                    break;
                    
                case LSTM_WB_O :
                    pthread_create(&thread_ids[i], NULL, forward_thread_ho, &args);
                    break;
                    
                case LSTM_WB_F :
                    pthread_create(&thread_ids[i], NULL, forward_thread_hf, &args);
                    break;
                    
                case LSTM_WB_I :
                    pthread_create(&thread_ids[i], NULL, forward_thread_hi, &args);
                    break;
                    
                default:
                    printf("\r\nInternal Error #101, Please report!\r\n");
                    exit(0);
                    break;
            }
        }
        for (i = 0; i < size; i++)
        {
            pthread_join(thread_ids[i], NULL);
        }
    }
    else
    {
        fully_connected_forward(cache_out->hf, model->W[LSTM_WB_F], X_one_hot, model->b[LSTM_WB_F], N, S);
        sigmoid_forward(cache_out->hf, cache_out->hf, N);

        fully_connected_forward(cache_out->hi, model->W[LSTM_WB_I], X_one_hot, model->b[LSTM_WB_I], N, S);
        sigmoid_forward(cache_out->hi, cache_out->hi, N);
        
        fully_connected_forward(cache_out->ho, model->W[LSTM_WB_O], X_one_hot, model->b[LSTM_WB_O], N, S);
        sigmoid_forward(cache_out->ho, cache_out->ho, N);

        fully_connected_forward(cache_out->hc, model->W[LSTM_WB_C], X_one_hot, model->b[LSTM_WB_C], N, S);
        if (model->params->use_tanf)
            tanf_forward(cache_out->hc, cache_out->hc, N);
        else
            tanh_forward(cache_out->hc, cache_out->hc, N);
    }
    
    // c = hf * c_old + hi * hc
    vectors_copy_multiply(cache_out->c, cache_out->hf, c_old, N);
    vectors_copy_multiply(tmp, cache_out->hc, cache_out->hi, N);
    vectors_add(cache_out->c, tmp, N);
    
    // h = ho * tanh_c_cache
    if (model->params->use_tanf)
        tanf_forward(cache_out->tanh_c_cache, cache_out->c, N);
    else
        tanh_forward(cache_out->tanh_c_cache, cache_out->c, N);
    
    vectors_copy_multiply(cache_out->h, cache_out->tanh_c_cache, cache_out->ho, N);
    
    // probs = softmax ( Wy*h + by )
    fully_connected_forward(cache_out->probs, model->W[LSTM_WB_Y], cache_out->h, model->b[LSTM_WB_Y], Y, N);
    if ( softmax > 0 )
    {
        softmax_layers_forward(cache_out->probs, cache_out->probs, Y, model->params->softmax_temp);
    }
#ifdef INTERLAYER_SIGMOID_ACTIVATION
    if ( softmax <= 0 ) {
        sigmoid_forward(cache_out->probs, cache_out->probs, Y);
        copy_vector(cache_out->probs_before_sigma, cache_out->probs, Y);
    }
#endif
    
    copy_vector(cache_out->X, X_one_hot, S);
    
#ifdef _WIN32
    free_vector(&tmp);
#endif
    
}



// model, input, state and cache values, &probs, whether or not to apply softmax
void lstm_forward_propagate(int layers,
                            numeric_t *first_layer_input,
                            lstm_model_t** model_layers,
                            lstm_values_cache_t ***caches_layer,
                            int e1, int e2)
{
    time_t prg_begin, prg_end;
    int p;
    struct timespec start, end;
      
    clock_gettime(CLOCK_MONOTONIC, &start);
    prg_begin = clock();

    for (p = layers-1; p >= 0; p--)
    {
        numeric_t * input = (p==layers-1) ? first_layer_input : caches_layer[p+1][e2]->probs;
        lstm_forward_propagate_layer(model_layers[p],
                                     input,
                                     caches_layer[p][e1],
                                     caches_layer[p][e2],
                                     p==0);
    }

    prg_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &end);
    performance.cpu_time_forward += (double)(prg_end - prg_begin) / (double)CLOCKS_PER_SEC;
    performance.time_forward += timespecDiff(&end, &start);
}

// A -= alpha * Am_hat / (np.sqrt(Rm_hat) + epsilon)
// Am_hat = Am / ( 1 - betaM ^ (iteration) )
// Rm_hat = Rm / ( 1 - betaR ^ (iteration) )

static void gradients_adam_optimizer(lstm_model_t* model, lstm_model_t* gradients, lstm_model_t* M, lstm_model_t* R, unsigned int t)
{
    numeric_t beta1 = model->params->beta1;
    numeric_t beta2 = model->params->beta2;
    
    numeric_t beta1t = 1.0 / ( 1.0 - pow(beta1, t+1));
    numeric_t beta2t = 1.0 / ( 1.0 - pow(beta2, t+1));
            
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_Y], gradients->W[LSTM_WB_Y], 1.0 - beta1, model->Y * model->N);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_I], gradients->W[LSTM_WB_I], 1.0 - beta1, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_C], gradients->W[LSTM_WB_C], 1.0 - beta1, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_O], gradients->W[LSTM_WB_O], 1.0 - beta1, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_F], gradients->W[LSTM_WB_F], 1.0 - beta1, model->N * model->S);
    
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_Y], gradients->b[LSTM_WB_Y], 1.0 - beta1, model->Y);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_I], gradients->b[LSTM_WB_I], 1.0 - beta1, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_C], gradients->b[LSTM_WB_C], 1.0 - beta1, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_O], gradients->b[LSTM_WB_O], 1.0 - beta1, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_F], gradients->b[LSTM_WB_F], 1.0 - beta1, model->N);
    
    vectors_multiply_scalar(M->W[LSTM_WB_Y], beta1, model->Y * model->N);
    vectors_multiply_scalar(M->W[LSTM_WB_I], beta1, model->N * model->S);
    vectors_multiply_scalar(M->W[LSTM_WB_C], beta1, model->N * model->S);
    vectors_multiply_scalar(M->W[LSTM_WB_O], beta1, model->N * model->S);
    vectors_multiply_scalar(M->W[LSTM_WB_F], beta1, model->N * model->S);
    
    vectors_multiply_scalar(M->b[LSTM_WB_Y], beta1, model->Y);
    vectors_multiply_scalar(M->b[LSTM_WB_I], beta1, model->N);
    vectors_multiply_scalar(M->b[LSTM_WB_C], beta1, model->N);
    vectors_multiply_scalar(M->b[LSTM_WB_O], beta1, model->N);
    vectors_multiply_scalar(M->b[LSTM_WB_F], beta1, model->N);
    
    vectors_add(M->W[LSTM_WB_Y], gradients->W[LSTM_WB_Y], model->Y * model->N);
    vectors_add(M->W[LSTM_WB_I], gradients->W[LSTM_WB_I], model->N * model->S);
    vectors_add(M->W[LSTM_WB_C], gradients->W[LSTM_WB_C], model->N * model->S);
    vectors_add(M->W[LSTM_WB_O], gradients->W[LSTM_WB_O], model->N * model->S);
    vectors_add(M->W[LSTM_WB_F], gradients->W[LSTM_WB_F], model->N * model->S);
    
    vectors_add(M->b[LSTM_WB_Y], gradients->b[LSTM_WB_Y], model->Y);
    vectors_add(M->b[LSTM_WB_I], gradients->b[LSTM_WB_I], model->N);
    vectors_add(M->b[LSTM_WB_C], gradients->b[LSTM_WB_C], model->N);
    vectors_add(M->b[LSTM_WB_O], gradients->b[LSTM_WB_O], model->N);
    vectors_add(M->b[LSTM_WB_F], gradients->b[LSTM_WB_F], model->N);
    
    // M Done!
    // Computing R
    
    vectors_multiply(gradients->W[LSTM_WB_Y], gradients->W[LSTM_WB_Y], model->Y * model->N);
    vectors_multiply(gradients->W[LSTM_WB_I], gradients->W[LSTM_WB_I], model->N * model->S);
    vectors_multiply(gradients->W[LSTM_WB_C], gradients->W[LSTM_WB_C], model->N * model->S);
    vectors_multiply(gradients->W[LSTM_WB_O], gradients->W[LSTM_WB_O], model->N * model->S);
    vectors_multiply(gradients->W[LSTM_WB_F], gradients->W[LSTM_WB_F], model->N * model->S);
    
    vectors_multiply(gradients->b[LSTM_WB_Y], gradients->b[LSTM_WB_Y], model->Y );
    vectors_multiply(gradients->b[LSTM_WB_I], gradients->b[LSTM_WB_I], model->N );
    vectors_multiply(gradients->b[LSTM_WB_C], gradients->b[LSTM_WB_C], model->N );
    vectors_multiply(gradients->b[LSTM_WB_O], gradients->b[LSTM_WB_O], model->N );
    vectors_multiply(gradients->b[LSTM_WB_F], gradients->b[LSTM_WB_F], model->N );
    
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_Y], gradients->W[LSTM_WB_Y], 1.0 - beta2, model->Y * model->N);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_I], gradients->W[LSTM_WB_I], 1.0 - beta2, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_C], gradients->W[LSTM_WB_C], 1.0 - beta2, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_O], gradients->W[LSTM_WB_O], 1.0 - beta2, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_F], gradients->W[LSTM_WB_F], 1.0 - beta2, model->N * model->S);
    
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_Y], gradients->b[LSTM_WB_Y], 1.0 - beta2, model->Y);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_I], gradients->b[LSTM_WB_I], 1.0 - beta2, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_C], gradients->b[LSTM_WB_C], 1.0 - beta2, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_O], gradients->b[LSTM_WB_O], 1.0 - beta2, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_F], gradients->b[LSTM_WB_F], 1.0 - beta2, model->N);
    
    vectors_multiply_scalar(R->W[LSTM_WB_Y], beta2, model->Y * model->N);
    vectors_multiply_scalar(R->W[LSTM_WB_I], beta2, model->N * model->S);
    vectors_multiply_scalar(R->W[LSTM_WB_C], beta2, model->N * model->S);
    vectors_multiply_scalar(R->W[LSTM_WB_O], beta2, model->N * model->S);
    vectors_multiply_scalar(R->W[LSTM_WB_F], beta2, model->N * model->S);
    
    vectors_multiply_scalar(R->b[LSTM_WB_Y], beta2, model->Y);
    vectors_multiply_scalar(R->b[LSTM_WB_I], beta2, model->N);
    vectors_multiply_scalar(R->b[LSTM_WB_C], beta2, model->N);
    vectors_multiply_scalar(R->b[LSTM_WB_O], beta2, model->N);
    vectors_multiply_scalar(R->b[LSTM_WB_F], beta2, model->N);
    
    vectors_add(R->W[LSTM_WB_Y], gradients->W[LSTM_WB_Y], model->Y * model->N);
    vectors_add(R->W[LSTM_WB_I], gradients->W[LSTM_WB_I], model->N * model->S);
    vectors_add(R->W[LSTM_WB_C], gradients->W[LSTM_WB_C], model->N * model->S);
    vectors_add(R->W[LSTM_WB_O], gradients->W[LSTM_WB_O], model->N * model->S);
    vectors_add(R->W[LSTM_WB_F], gradients->W[LSTM_WB_F], model->N * model->S);
    
    vectors_add(R->b[LSTM_WB_Y], gradients->b[LSTM_WB_Y], model->Y);
    vectors_add(R->b[LSTM_WB_I], gradients->b[LSTM_WB_I], model->N);
    vectors_add(R->b[LSTM_WB_C], gradients->b[LSTM_WB_C], model->N);
    vectors_add(R->b[LSTM_WB_O], gradients->b[LSTM_WB_O], model->N);
    vectors_add(R->b[LSTM_WB_F], gradients->b[LSTM_WB_F], model->N);
    
    // R done!
        
    vectors_copy_multiply_scalar(M->Wm[LSTM_WB_Y], M->W[LSTM_WB_Y], beta1t, model->Y * model->N);
    vectors_copy_multiply_scalar(M->Wm[LSTM_WB_I], M->W[LSTM_WB_I], beta1t, model->N * model->S);
    vectors_copy_multiply_scalar(M->Wm[LSTM_WB_C], M->W[LSTM_WB_C], beta1t, model->N * model->S);
    vectors_copy_multiply_scalar(M->Wm[LSTM_WB_O], M->W[LSTM_WB_O], beta1t, model->N * model->S);
    vectors_copy_multiply_scalar(M->Wm[LSTM_WB_F], M->W[LSTM_WB_F], beta1t, model->N * model->S);
    
    vectors_copy_multiply_scalar(M->bm[LSTM_WB_Y], M->b[LSTM_WB_Y], beta1t, model->Y);
    vectors_copy_multiply_scalar(M->bm[LSTM_WB_I], M->b[LSTM_WB_I], beta1t, model->N);
    vectors_copy_multiply_scalar(M->bm[LSTM_WB_C], M->b[LSTM_WB_C], beta1t, model->N);
    vectors_copy_multiply_scalar(M->bm[LSTM_WB_O], M->b[LSTM_WB_O], beta1t, model->N);
    vectors_copy_multiply_scalar(M->bm[LSTM_WB_F], M->b[LSTM_WB_F], beta1t, model->N);
    
    // M hat done!
    
    vectors_copy_multiply_scalar(R->Wm[LSTM_WB_Y], R->W[LSTM_WB_Y], beta2t, model->Y * model->N);
    vectors_copy_multiply_scalar(R->Wm[LSTM_WB_I], R->W[LSTM_WB_I], beta2t, model->N * model->S);
    vectors_copy_multiply_scalar(R->Wm[LSTM_WB_C], R->W[LSTM_WB_C], beta2t, model->N * model->S);
    vectors_copy_multiply_scalar(R->Wm[LSTM_WB_O], R->W[LSTM_WB_O], beta2t, model->N * model->S);
    vectors_copy_multiply_scalar(R->Wm[LSTM_WB_F], R->W[LSTM_WB_F], beta2t, model->N * model->S);
    
    vectors_copy_multiply_scalar(R->bm[LSTM_WB_Y], R->b[LSTM_WB_Y], beta2t, model->Y);
    vectors_copy_multiply_scalar(R->bm[LSTM_WB_I], R->b[LSTM_WB_I], beta2t, model->N);
    vectors_copy_multiply_scalar(R->bm[LSTM_WB_C], R->b[LSTM_WB_C], beta2t, model->N);
    vectors_copy_multiply_scalar(R->bm[LSTM_WB_O], R->b[LSTM_WB_O], beta2t, model->N);
    vectors_copy_multiply_scalar(R->bm[LSTM_WB_F], R->b[LSTM_WB_F], beta2t, model->N);
    
    // R hat done!
    
    vector_sqrt(R->Wm[LSTM_WB_Y], model->Y * model->N);
    vector_sqrt(R->Wm[LSTM_WB_I], model->N * model->S);
    vector_sqrt(R->Wm[LSTM_WB_C], model->N * model->S);
    vector_sqrt(R->Wm[LSTM_WB_O], model->N * model->S);
    vector_sqrt(R->Wm[LSTM_WB_F], model->N * model->S);
    
    vector_sqrt(R->bm[LSTM_WB_Y], model->Y);
    vector_sqrt(R->bm[LSTM_WB_I], model->N);
    vector_sqrt(R->bm[LSTM_WB_C], model->N);
    vector_sqrt(R->bm[LSTM_WB_O], model->N);
    vector_sqrt(R->bm[LSTM_WB_F], model->N);
    
    vectors_add_scalar(R->Wm[LSTM_WB_Y], EPSILON, model->Y * model->N);
    vectors_add_scalar(R->Wm[LSTM_WB_I], EPSILON, model->N * model->S);
    vectors_add_scalar(R->Wm[LSTM_WB_C], EPSILON, model->N * model->S);
    vectors_add_scalar(R->Wm[LSTM_WB_O], EPSILON, model->N * model->S);
    vectors_add_scalar(R->Wm[LSTM_WB_F], EPSILON, model->N * model->S);
    
    vectors_add_scalar(R->bm[LSTM_WB_Y], EPSILON, model->Y);
    vectors_add_scalar(R->bm[LSTM_WB_I], EPSILON, model->N);
    vectors_add_scalar(R->bm[LSTM_WB_C], EPSILON, model->N);
    vectors_add_scalar(R->bm[LSTM_WB_O], EPSILON, model->N);
    vectors_add_scalar(R->bm[LSTM_WB_F], EPSILON, model->N);
        
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_Y], M->Wm[LSTM_WB_Y], model->params->learning_rate, model->Y * model->N);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_I], M->Wm[LSTM_WB_I], model->params->learning_rate, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_C], M->Wm[LSTM_WB_C], model->params->learning_rate, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_O], M->Wm[LSTM_WB_O], model->params->learning_rate, model->N * model->S);
    vectors_copy_multiply_scalar(gradients->Wm[LSTM_WB_F], M->Wm[LSTM_WB_F], model->params->learning_rate, model->N * model->S);
    
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_Y], M->bm[LSTM_WB_Y], model->params->learning_rate, model->Y);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_I], M->bm[LSTM_WB_I], model->params->learning_rate, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_C], M->bm[LSTM_WB_C], model->params->learning_rate, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_O], M->bm[LSTM_WB_O], model->params->learning_rate, model->N);
    vectors_copy_multiply_scalar(gradients->bm[LSTM_WB_F], M->bm[LSTM_WB_F], model->params->learning_rate, model->N);
    
    vectors_div(gradients->Wm[LSTM_WB_Y], R->Wm[LSTM_WB_Y], model->Y * model->N);
    vectors_div(gradients->Wm[LSTM_WB_I], R->Wm[LSTM_WB_I], model->N * model->S);
    vectors_div(gradients->Wm[LSTM_WB_C], R->Wm[LSTM_WB_C], model->N * model->S);
    vectors_div(gradients->Wm[LSTM_WB_O], R->Wm[LSTM_WB_O], model->N * model->S);
    vectors_div(gradients->Wm[LSTM_WB_F], R->Wm[LSTM_WB_F], model->N * model->S);
    
    vectors_div(gradients->bm[LSTM_WB_Y], R->bm[LSTM_WB_Y], model->Y);
    vectors_div(gradients->bm[LSTM_WB_I], R->bm[LSTM_WB_I], model->N);
    vectors_div(gradients->bm[LSTM_WB_C], R->bm[LSTM_WB_C], model->N);
    vectors_div(gradients->bm[LSTM_WB_O], R->bm[LSTM_WB_O], model->N);
    vectors_div(gradients->bm[LSTM_WB_F], R->bm[LSTM_WB_F], model->N);
    
    vectors_subtract(model->W[LSTM_WB_Y], gradients->Wm[LSTM_WB_Y], model->Y * model->N);
    vectors_subtract(model->W[LSTM_WB_I], gradients->Wm[LSTM_WB_I], model->N * model->S);
    vectors_subtract(model->W[LSTM_WB_C], gradients->Wm[LSTM_WB_C], model->N * model->S);
    vectors_subtract(model->W[LSTM_WB_O], gradients->Wm[LSTM_WB_O], model->N * model->S);
    vectors_subtract(model->W[LSTM_WB_F], gradients->Wm[LSTM_WB_F], model->N * model->S);
    
    vectors_subtract(model->b[LSTM_WB_Y], gradients->bm[LSTM_WB_Y], model->Y);
    vectors_subtract(model->b[LSTM_WB_I], gradients->bm[LSTM_WB_I], model->N);
    vectors_subtract(model->b[LSTM_WB_C], gradients->bm[LSTM_WB_C], model->N);
    vectors_subtract(model->b[LSTM_WB_O], gradients->bm[LSTM_WB_O], model->N);
    vectors_subtract(model->b[LSTM_WB_F], gradients->bm[LSTM_WB_F], model->N);
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

static void * optimize_thread(void * vargp)
{
    thread_arg_t * arg = (thread_arg_t *)vargp;
    adam_optimize_layer(arg->model,
                        arg->gradients,
                        arg->M,
                        arg->R,
                        arg->t,
                        arg->use_thread);
    return NULL;
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

        for (p = 0; p < size; p++)
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
