//
//  lstm.c
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/10/22.
//

#include "lstm-mt.h"

void lstm_init_fail(const char * msg)
{
    printf("%s: %s",__func__,msg);
    exit(-1);
}

// Inputs, Neurons, Outputs, &lstm model, zeros
int lstm_init_model(int X, int N, int Y,
                    lstm_model_t **model_to_be_set, int zeros,
                    lstm_model_parameters_t *params)
{
    int S = X + N;
    lstm_model_t* lstm = e_calloc(1, sizeof(lstm_model_t));
    
    lstm->X = X;
    lstm->N = N;
    lstm->S = S;
    lstm->Y = Y;
    
    lstm->params = params;
    
    if ( zeros ) 
    {
        lstm->W[LSTM_WB_F] = get_zero_vector(N * S);
        lstm->W[LSTM_WB_I] = get_zero_vector(N * S);
        lstm->W[LSTM_WB_C] = get_zero_vector(N * S);
        lstm->W[LSTM_WB_O] = get_zero_vector(N * S);
        lstm->W[LSTM_WB_Y] = get_zero_vector(Y * N);
    }
    else 
    {
        lstm->W[LSTM_WB_F] = get_random_vector(N * S, S);
        lstm->W[LSTM_WB_I] = get_random_vector(N * S, S);
        lstm->W[LSTM_WB_C] = get_random_vector(N * S, S);
        lstm->W[LSTM_WB_O] = get_random_vector(N * S, S);
        lstm->W[LSTM_WB_Y] = get_random_vector(Y * N, N);
    }
    
    lstm->b[LSTM_WB_F] = get_zero_vector(N);
    lstm->b[LSTM_WB_I] = get_zero_vector(N);
    lstm->b[LSTM_WB_C] = get_zero_vector(N);
    lstm->b[LSTM_WB_O] = get_zero_vector(N);
    lstm->b[LSTM_WB_Y] = get_zero_vector(Y);
    
    lstm->dldhf = get_zero_vector(N);
    lstm->dldhi = get_zero_vector(N);
    lstm->dldhc = get_zero_vector(N);
    lstm->dldho = get_zero_vector(N);
    lstm->dldc  = get_zero_vector(N);
    lstm->dldh  = get_zero_vector(N);
    
    lstm->dldXc = get_zero_vector(S);
    lstm->dldXo = get_zero_vector(S);
    lstm->dldXi = get_zero_vector(S);
    lstm->dldXf = get_zero_vector(S);
    
    // Gradient descent momentum caches
    lstm->Wm[LSTM_WB_F] = get_zero_vector(N * S);
    lstm->Wm[LSTM_WB_I] = get_zero_vector(N * S);
    lstm->Wm[LSTM_WB_C] = get_zero_vector(N * S);
    lstm->Wm[LSTM_WB_O] = get_zero_vector(N * S);
    lstm->Wm[LSTM_WB_Y] = get_zero_vector(Y * N);
    
    lstm->bm[LSTM_WB_F] = get_zero_vector(N);
    lstm->bm[LSTM_WB_I] = get_zero_vector(N);
    lstm->bm[LSTM_WB_C] = get_zero_vector(N);
    lstm->bm[LSTM_WB_O] = get_zero_vector(N);
    lstm->bm[LSTM_WB_Y] = get_zero_vector(Y);
    
    *model_to_be_set = lstm;
    
    return 0;
}
// lstm model to be freed
void lstm_free_model(lstm_model_t* lstm)
{
    free_vector(&lstm->W[LSTM_WB_F]);
    free_vector(&lstm->W[LSTM_WB_I]);
    free_vector(&lstm->W[LSTM_WB_C]);
    free_vector(&lstm->W[LSTM_WB_O]);
    free_vector(&lstm->W[LSTM_WB_Y]);
    
    free_vector(&lstm->b[LSTM_WB_F]);
    free_vector(&lstm->b[LSTM_WB_I]);
    free_vector(&lstm->b[LSTM_WB_C]);
    free_vector(&lstm->b[LSTM_WB_O]);
    free_vector(&lstm->b[LSTM_WB_Y]);
    
    free_vector(&lstm->dldhf);
    free_vector(&lstm->dldhi);
    free_vector(&lstm->dldhc);
    free_vector(&lstm->dldho);
    free_vector(&lstm->dldc);
    free_vector(&lstm->dldh);
    
    free_vector(&lstm->dldXc);
    free_vector(&lstm->dldXo);
    free_vector(&lstm->dldXi);
    free_vector(&lstm->dldXf);
    
    free_vector(&lstm->Wm[LSTM_WB_F]);
    free_vector(&lstm->Wm[LSTM_WB_I]);
    free_vector(&lstm->Wm[LSTM_WB_C]);
    free_vector(&lstm->Wm[LSTM_WB_O]);
    free_vector(&lstm->Wm[LSTM_WB_Y]);
    
    free_vector(&lstm->bm[LSTM_WB_F]);
    free_vector(&lstm->bm[LSTM_WB_I]);
    free_vector(&lstm->bm[LSTM_WB_C]);
    free_vector(&lstm->bm[LSTM_WB_O]);
    free_vector(&lstm->bm[LSTM_WB_Y]);
    
    free(lstm);
}

void lstm_cache_container_free(lstm_values_cache_t* cache_to_be_freed)
{
    free_vector(&(cache_to_be_freed)->probs);
    free_vector(&(cache_to_be_freed)->probs_before_sigma);
    free_vector(&(cache_to_be_freed)->c);
    free_vector(&(cache_to_be_freed)->h);
    free_vector(&(cache_to_be_freed)->c_old);
    free_vector(&(cache_to_be_freed)->h_old);
    free_vector(&(cache_to_be_freed)->X);
    free_vector(&(cache_to_be_freed)->hf);
    free_vector(&(cache_to_be_freed)->hi);
    free_vector(&(cache_to_be_freed)->ho);
    free_vector(&(cache_to_be_freed)->hc);
    free_vector(&(cache_to_be_freed)->tanh_c_cache);
}

lstm_values_cache_t*  lstm_cache_container_init(int X, int N, int Y)
{
    int S = N + X;
    
    lstm_values_cache_t* cache = e_calloc(1, sizeof(lstm_values_cache_t));
    
    cache->probs = get_zero_vector(Y);
    cache->probs_before_sigma = get_zero_vector(Y);
    cache->c = get_zero_vector(N);
    cache->h = get_zero_vector(N);
    cache->c_old = get_zero_vector(N);
    cache->h_old = get_zero_vector(N);
    cache->X = get_zero_vector(S);
    cache->hf = get_zero_vector(N);
    cache->hi = get_zero_vector(N);
    cache->ho = get_zero_vector(N);
    cache->hc = get_zero_vector(N);
    cache->tanh_c_cache = get_zero_vector(N);
    
    return cache;
}

void lstm_values_state_init(lstm_values_state_t** d_next_to_set, int N)
{
    lstm_values_state_t * d_next = e_calloc(1, sizeof(lstm_values_state_t));
    
    init_zero_vector(&d_next->c, N);
    init_zero_vector(&d_next->h, N);
    
    *d_next_to_set = d_next;
}

int gradients_fit(lstm_model_t* gradients, numeric_t limit)
{
    int msg = 0;
    msg += vectors_fit(gradients->W[LSTM_WB_Y], limit, gradients->Y * gradients->N);
    msg += vectors_fit(gradients->W[LSTM_WB_I], limit, gradients->N * gradients->S);
    msg += vectors_fit(gradients->W[LSTM_WB_C], limit, gradients->N * gradients->S);
    msg += vectors_fit(gradients->W[LSTM_WB_O], limit, gradients->N * gradients->S);
    msg += vectors_fit(gradients->W[LSTM_WB_F], limit, gradients->N * gradients->S);
    
    msg += vectors_fit(gradients->b[LSTM_WB_Y], limit, gradients->Y);
    msg += vectors_fit(gradients->b[LSTM_WB_I], limit, gradients->N);
    msg += vectors_fit(gradients->b[LSTM_WB_C], limit, gradients->N);
    msg += vectors_fit(gradients->b[LSTM_WB_F], limit, gradients->N);
    msg += vectors_fit(gradients->b[LSTM_WB_O], limit, gradients->N);
    
    return msg;
}

int gradients_clip(lstm_model_t* gradients, numeric_t limit)
{
    int msg = 0;
    msg += vectors_clip(gradients->W[LSTM_WB_Y], limit, gradients->Y * gradients->N);
    msg += vectors_clip(gradients->W[LSTM_WB_I], limit, gradients->N * gradients->S);
    msg += vectors_clip(gradients->W[LSTM_WB_C], limit, gradients->N * gradients->S);
    msg += vectors_clip(gradients->W[LSTM_WB_O], limit, gradients->N * gradients->S);
    msg += vectors_clip(gradients->W[LSTM_WB_F], limit, gradients->N * gradients->S);
    
    msg += vectors_clip(gradients->b[LSTM_WB_Y], limit, gradients->Y);
    msg += vectors_clip(gradients->b[LSTM_WB_I], limit, gradients->N);
    msg += vectors_clip(gradients->b[LSTM_WB_C], limit, gradients->N);
    msg += vectors_clip(gradients->b[LSTM_WB_F], limit, gradients->N);
    msg += vectors_clip(gradients->b[LSTM_WB_O], limit, gradients->N);
    
    return msg;
}

void sum_gradients(lstm_model_t* gradients, lstm_model_t* gradients_entry)
{
    vectors_add(gradients->W[LSTM_WB_Y], gradients_entry->W[LSTM_WB_Y], gradients->Y * gradients->N);
    vectors_add(gradients->W[LSTM_WB_I], gradients_entry->W[LSTM_WB_I], gradients->N * gradients->S);
    vectors_add(gradients->W[LSTM_WB_C], gradients_entry->W[LSTM_WB_C], gradients->N * gradients->S);
    vectors_add(gradients->W[LSTM_WB_O], gradients_entry->W[LSTM_WB_O], gradients->N * gradients->S);
    vectors_add(gradients->W[LSTM_WB_F], gradients_entry->W[LSTM_WB_F], gradients->N * gradients->S);
    
    vectors_add(gradients->b[LSTM_WB_Y], gradients_entry->b[LSTM_WB_Y], gradients->Y);
    vectors_add(gradients->b[LSTM_WB_I], gradients_entry->b[LSTM_WB_I], gradients->N);
    vectors_add(gradients->b[LSTM_WB_C], gradients_entry->b[LSTM_WB_C], gradients->N);
    vectors_add(gradients->b[LSTM_WB_F], gradients_entry->b[LSTM_WB_F], gradients->N);
    vectors_add(gradients->b[LSTM_WB_O], gradients_entry->b[LSTM_WB_O], gradients->N);
}

// A = A - alpha * m, m = momentum * m + ( 1 - momentum ) * dldA
void gradients_decend(lstm_model_t* model, lstm_model_t* gradients)
{
    // Computing momentum * m
    vectors_multiply_scalar(gradients->Wm[LSTM_WB_Y], model->params->momentum, model->Y * model->N);
    vectors_multiply_scalar(gradients->Wm[LSTM_WB_I], model->params->momentum, model->N * model->S);
    vectors_multiply_scalar(gradients->Wm[LSTM_WB_C], model->params->momentum, model->N * model->S);
    vectors_multiply_scalar(gradients->Wm[LSTM_WB_O], model->params->momentum, model->N * model->S);
    vectors_multiply_scalar(gradients->Wm[LSTM_WB_F], model->params->momentum, model->N * model->S);
    
    vectors_multiply_scalar(gradients->bm[LSTM_WB_Y], model->params->momentum, model->Y);
    vectors_multiply_scalar(gradients->bm[LSTM_WB_I], model->params->momentum, model->N);
    vectors_multiply_scalar(gradients->bm[LSTM_WB_C], model->params->momentum, model->N);

    vectors_multiply_scalar(gradients->bm[LSTM_WB_O], model->params->momentum, model->N);
    vectors_multiply_scalar(gradients->bm[LSTM_WB_F], model->params->momentum, model->N);
    
    // Computing m = momentum * m + (1 - momentum) * dldA
    vectors_add_scalar_multiply(gradients->Wm[LSTM_WB_Y], gradients->W[LSTM_WB_Y], model->Y * model->N, 1.0 - model->params->momentum);
    vectors_add_scalar_multiply(gradients->Wm[LSTM_WB_I], gradients->W[LSTM_WB_I], model->N * model->S, 1.0 - model->params->momentum);
    vectors_add_scalar_multiply(gradients->Wm[LSTM_WB_C], gradients->W[LSTM_WB_C], model->N * model->S, 1.0 - model->params->momentum);
    vectors_add_scalar_multiply(gradients->Wm[LSTM_WB_O], gradients->W[LSTM_WB_O], model->N * model->S, 1.0 - model->params->momentum);
    vectors_add_scalar_multiply(gradients->Wm[LSTM_WB_F], gradients->W[LSTM_WB_F], model->N * model->S, 1.0 - model->params->momentum);
    
    vectors_add_scalar_multiply(gradients->bm[LSTM_WB_Y], gradients->b[LSTM_WB_Y], model->Y, 1.0 - model->params->momentum);
    vectors_add_scalar_multiply(gradients->bm[LSTM_WB_I], gradients->b[LSTM_WB_I], model->N, 1.0 - model->params->momentum);
    vectors_add_scalar_multiply(gradients->bm[LSTM_WB_C], gradients->b[LSTM_WB_C], model->N, 1.0 - model->params->momentum);
    vectors_add_scalar_multiply(gradients->bm[LSTM_WB_O], gradients->b[LSTM_WB_O], model->N, 1.0 - model->params->momentum);
    vectors_add_scalar_multiply(gradients->bm[LSTM_WB_F], gradients->b[LSTM_WB_F], model->N, 1.0 - model->params->momentum);
    
    // Computing A = A - alpha * m
    vectors_subtract_scalar_multiply(model->W[LSTM_WB_Y], gradients->Wm[LSTM_WB_Y], model->Y * model->N, model->params->learning_rate);
    vectors_subtract_scalar_multiply(model->W[LSTM_WB_I], gradients->Wm[LSTM_WB_I], model->N * model->S, model->params->learning_rate);
    vectors_subtract_scalar_multiply(model->W[LSTM_WB_C], gradients->Wm[LSTM_WB_C], model->N * model->S, model->params->learning_rate);
    vectors_subtract_scalar_multiply(model->W[LSTM_WB_O], gradients->Wm[LSTM_WB_O], model->N * model->S, model->params->learning_rate);
    vectors_subtract_scalar_multiply(model->W[LSTM_WB_F], gradients->Wm[LSTM_WB_F], model->N * model->S, model->params->learning_rate);
    
    vectors_subtract_scalar_multiply(model->b[LSTM_WB_Y], gradients->bm[LSTM_WB_Y], model->Y, model->params->learning_rate);
    vectors_subtract_scalar_multiply(model->b[LSTM_WB_I], gradients->bm[LSTM_WB_I], model->N, model->params->learning_rate);
    vectors_subtract_scalar_multiply(model->b[LSTM_WB_C], gradients->bm[LSTM_WB_C], model->N, model->params->learning_rate);
    vectors_subtract_scalar_multiply(model->b[LSTM_WB_F], gradients->bm[LSTM_WB_F], model->N, model->params->learning_rate);
    vectors_subtract_scalar_multiply(model->b[LSTM_WB_O], gradients->bm[LSTM_WB_O], model->N, model->params->learning_rate);
}

void lstm_values_next_cache_init(lstm_values_next_cache_t** d_next_to_set, int N, int X)
{
    lstm_values_next_cache_t * d_next = e_calloc(1, sizeof(lstm_values_next_cache_t));
    
    init_zero_vector(&d_next->dldh_next, N);
    init_zero_vector(&d_next->dldc_next, N);
    init_zero_vector(&d_next->dldY_pass, X);
    *d_next_to_set = d_next;
}
void lstm_values_next_cache_free(lstm_values_next_cache_t* d_next)
{
    free_vector(&d_next->dldc_next);
    free_vector(&d_next->dldh_next);
    free_vector(&d_next->dldY_pass);
    free(d_next);
}

void lstm_values_next_state_free(lstm_values_state_t* d_next)
{
    free_vector(&d_next->h);
    free_vector(&d_next->c);
    free(d_next);
}

void lstm_zero_the_model(lstm_model_t * model)
{
    vector_set_to_zero(model->W[LSTM_WB_Y], model->Y * model->N);
    vector_set_to_zero(model->W[LSTM_WB_I], model->N * model->S);
    vector_set_to_zero(model->W[LSTM_WB_C], model->N * model->S);
    vector_set_to_zero(model->W[LSTM_WB_O], model->N * model->S);
    vector_set_to_zero(model->W[LSTM_WB_F], model->N * model->S);
    
    vector_set_to_zero(model->b[LSTM_WB_Y], model->Y);
    vector_set_to_zero(model->b[LSTM_WB_I], model->N);
    vector_set_to_zero(model->b[LSTM_WB_C], model->N);
    vector_set_to_zero(model->b[LSTM_WB_F], model->N);
    vector_set_to_zero(model->b[LSTM_WB_O], model->N);
    
    vector_set_to_zero(model->Wm[LSTM_WB_Y], model->Y * model->N);
    vector_set_to_zero(model->Wm[LSTM_WB_I], model->N * model->S);
    vector_set_to_zero(model->Wm[LSTM_WB_C], model->N * model->S);
    vector_set_to_zero(model->Wm[LSTM_WB_O], model->N * model->S);
    vector_set_to_zero(model->Wm[LSTM_WB_F], model->N * model->S);
    
    vector_set_to_zero(model->bm[LSTM_WB_Y], model->Y);
    vector_set_to_zero(model->bm[LSTM_WB_I], model->N);
    vector_set_to_zero(model->bm[LSTM_WB_C], model->N);
    vector_set_to_zero(model->bm[LSTM_WB_F], model->N);
    vector_set_to_zero(model->bm[LSTM_WB_O], model->N);
    
    vector_set_to_zero(model->dldhf, model->N);
    vector_set_to_zero(model->dldhi, model->N);
    vector_set_to_zero(model->dldhc, model->N);
    vector_set_to_zero(model->dldho, model->N);
    vector_set_to_zero(model->dldc, model->N);
    vector_set_to_zero(model->dldh, model->N);
    
    vector_set_to_zero(model->dldXc, model->S);
    vector_set_to_zero(model->dldXo, model->S);
    vector_set_to_zero(model->dldXi, model->S);
    vector_set_to_zero(model->dldXf, model->S);
}

void lstm_zero_d_next(lstm_values_next_cache_t * d_next,
                      int inputs, int neurons)
{
    vector_set_to_zero(d_next->dldh_next, neurons);
    vector_set_to_zero(d_next->dldc_next, neurons);
    vector_set_to_zero(d_next->dldY_pass, inputs);
}

void lstm_next_state_copy(lstm_values_state_t * state, lstm_values_cache_t * cache, int neurons, int write)
{
    if ( write )
    {
        // Write to the state carrying unit
        copy_vector(state->h, cache->h, neurons);
        copy_vector(state->c, cache->c, neurons);
    }
    else
    {
        // Withdraw from the state carrying unit
        copy_vector(cache->h, state->h, neurons);
        copy_vector(cache->c, state->c, neurons);
    }
}

void lstm_cache_container_set_start(lstm_values_cache_t * cache, int neurons)
{
    // State variables set to zero
    vector_set_to_zero(cache->h, neurons);
    vector_set_to_zero(cache->c, neurons);
    
}

void lstm_store_net_layers(lstm_model_t** model, FILE *fp, unsigned int layers)
{
    unsigned int p;    
    for (p = 0; p < layers; p++) 
    {        
#ifdef STORE_NET_AS_ASCII
        vector_store_ascii(model[p]->W[LSTM_WB_Y], model[p]->Y * model[p]->N, fp);
        vector_store_ascii(model[p]->W[LSTM_WB_I], model[p]->N * model[p]->S, fp);
        vector_store_ascii(model[p]->W[LSTM_WB_C], model[p]->N * model[p]->S, fp);
        vector_store_ascii(model[p]->W[LSTM_WB_O], model[p]->N * model[p]->S, fp);
        vector_store_ascii(model[p]->W[LSTM_WB_F], model[p]->N * model[p]->S, fp);
        
        vector_store_ascii(model[p]->b[LSTM_WB_Y], model[p]->Y, fp);
        vector_store_ascii(model[p]->b[LSTM_WB_I], model[p]->N, fp);
        vector_store_ascii(model[p]->b[LSTM_WB_C], model[p]->N, fp);
        vector_store_ascii(model[p]->b[LSTM_WB_F], model[p]->N, fp);
        vector_store_ascii(model[p]->b[LSTM_WB_O], model[p]->N, fp);
#else
        vector_store(model[p]->W[LSTM_WB_Y], model[p]->Y * model[p]->N, fp);
        vector_store(model[p]->W[LSTM_WB_I], model[p]->N * model[p]->S, fp);
        vector_store(model[p]->W[LSTM_WB_C], model[p]->N * model[p]->S, fp);
        vector_store(model[p]->W[LSTM_WB_O], model[p]->N * model[p]->S, fp);
        vector_store(model[p]->W[LSTM_WB_F], model[p]->N * model[p]->S, fp);
        
        vector_store(model[p]->b[LSTM_WB_Y], model[p]->Y, fp);
        vector_store(model[p]->b[LSTM_WB_I], model[p]->N, fp);
        vector_store(model[p]->b[LSTM_WB_C], model[p]->N, fp);
        vector_store(model[p]->b[LSTM_WB_F], model[p]->N, fp);
        vector_store(model[p]->b[LSTM_WB_O], model[p]->N, fp);
#endif
    }
}

void lstm_store_net_layers_as_json(lstm_model_t** model, const char * filename,
                                   const char *set_name, set_t *set, unsigned int layers)
{
    FILE * fp;
    unsigned int p = 0;
    
    fp = fopen(filename, "w");
    
    if ( fp == NULL ) 
    {
        printf("Failed to open file: %s for writing.\n", filename);
        return;
    }
    
    fprintf(fp, "{\n\"%s\": ", set_name);
    set_store_as_json(set, fp);
    
    fprintf(fp, ",\n\"LSTM layers\": %d,\n", layers);
    
    while ( p < layers ) 
    {        
        if ( p > 0 )
            fprintf(fp, ",\n");
        
        fprintf(fp, "\"Layer %d\": {\n", p+1);
        
        fprintf(fp, "\t\"Wy\": ");
        vector_store_as_matrix_json(model[p]->W[LSTM_WB_Y], model[p]->Y, model[p]->N, fp);
        fprintf(fp, ",\n\t\"Wi\": ");
        vector_store_as_matrix_json(model[p]->W[LSTM_WB_I], model[p]->N, model[p]->S, fp);
        fprintf(fp, ",\n\t\"Wc\": ");
        vector_store_as_matrix_json(model[p]->W[LSTM_WB_C], model[p]->N, model[p]->S, fp);
        fprintf(fp, ",\n\t\"Wo\": ");
        vector_store_as_matrix_json(model[p]->W[LSTM_WB_O], model[p]->N, model[p]->S, fp);
        fprintf(fp, ",\n\t\"Wf\": ");
        vector_store_as_matrix_json(model[p]->W[LSTM_WB_F], model[p]->N, model[p]->S, fp);
        
        fprintf(fp, ",\n\t\"by\": ");
        vector_store_json(model[p]->b[LSTM_WB_Y], model[p]->Y, fp);
        fprintf(fp, ",\n\t\"bi\": ");
        vector_store_json(model[p]->b[LSTM_WB_I], model[p]->N, fp);
        fprintf(fp, ",\n\t\"bc\": ");
        vector_store_json(model[p]->b[LSTM_WB_C], model[p]->N, fp);
        fprintf(fp, ",\n\t\"bf\": ");
        vector_store_json(model[p]->b[LSTM_WB_F], model[p]->N, fp);
        fprintf(fp, ",\n\t\"bo\": ");
        vector_store_json(model[p]->b[LSTM_WB_O], model[p]->N, fp);
        
        fprintf(fp, "}\n");
        
        ++p;
    }
    
    fprintf(fp, "}\n");
    
    fclose(fp);    
}

// Exits the program if EOF is encountered
static void e_lstm_fgets(char *str, int n, FILE *fp)
{
    if ( fgets(str, n, fp) == NULL )
    {
        fprintf(stderr, "lstm_read error: unexpected EOF. \
Net-file incompatible with current version.\n");
        fflush(stderr);
        exit(1);
    }
}

void lstm_load(const char *path, set_t *set,
               lstm_model_parameters_t *params, lstm_model_t ***model)
{
    FILE * fp;
    char intContainer[10];
    int f;
    int F;
    int L;
    int l;
    int layerInputs[LSTM_MAX_LAYERS];
    int layerNodes[LSTM_MAX_LAYERS];
    int layerOutputs[LSTM_MAX_LAYERS];
    int FileVersion;
    
    fp = fopen(path, "r");
    
    if ( fp == NULL ) 
    {
        printf("%s error: Failed to open file: %s for reading.\n", __func__, path);
        exit(1);
    }
    
    initialize_set(set);
    
    /*
     * LSTM net file structure
     * File version   BINARY_FILE_VERSION
     * NbrFeatures    (F)
     * NbrLayers      (L)
     * Nodes in layer 1 (output layer)
     * Nodes in layer 2
     * ...
     * Nodes in layer L (input layer)
     * Feature Value 1 (int in ASCII [0-255])
     * Feature Value 2
     * ...
     * Feature Value F
     * --- From here on it is a blob of bytes ---
     * Layer 1: Wy
     * Layer 1: Wi
     * Layer 1: Wc
     * Layer 1: Wo
     * Layer 1: Wf
     * Layer 1: by
     * Layer 1: bi
     * Layer 1: bc
     * Layer 1: bf
     * Layer 1: bo
     * ...
     * Layer L: Wy
     * Layer L: Wi
     * ...
     */
    
    // Read file version
    e_lstm_fgets(intContainer, sizeof(intContainer), fp);
    FileVersion = atoi(intContainer);
    (void) FileVersion; // Not used yet, in this early stage
    // Read NbrFeatures
    e_lstm_fgets(intContainer, sizeof(intContainer), fp);
    F = atoi(intContainer);
    
    // Read NbrLayers
    e_lstm_fgets(intContainer, sizeof(intContainer), fp);
    L = atoi(intContainer);
    
    if ( L > LSTM_MAX_LAYERS )
    {
        // This is too many layers
        fprintf(stderr, "%s error: Failed to load network, too many layers.\n", __func__);
        exit(1);
    }
    
    // Setting the number of layers among the parameters
    params->layers = L;
    
    l = 0;
    while ( l < L )
    {
        // Read number of inputs, nodes and ouput in this layer
        e_lstm_fgets(intContainer, sizeof(intContainer), fp);
        layerInputs[l] = atoi(intContainer);
        e_lstm_fgets(intContainer, sizeof(intContainer), fp);
        layerNodes[l] = atoi(intContainer);
        e_lstm_fgets(intContainer, sizeof(intContainer), fp);
        layerOutputs[l] = atoi(intContainer);
        ++l;
    }
    
    // Setting the number of neurons
    // NOTE: it is the same for each layer (for now)
    params->neurons = layerNodes[0];
    
    // Import feature set
    f = 0;
    while ( f < F )
    {
        e_lstm_fgets(intContainer, sizeof(intContainer), fp);
        set->values[f] = (char)atoi(intContainer);
        set->free[f] = 0;
        ++f;
    }
    
    assert(set_get_features(set) == layerInputs[L-1]);
    
    *model = (lstm_model_t**) malloc(L*sizeof(lstm_model_t*));
    if ( *model == NULL )
        lstm_init_fail("Failed to allocate resources for the net read\n");
    
    l = 0;
    while ( l < L )
    {
        lstm_init_model(
                        layerInputs[l],
                        layerNodes[l],
                        layerOutputs[l],
                        &(*model)[l], 0, params);
        ++l;
    }
    
    lstm_read_net_layers(*model, fp, L);
    
    fclose(fp);
}

void lstm_store(const char *path, set_t *set,
                lstm_model_t **model, unsigned int layers)
{
    FILE * fp;
    int f;
    int F = set_get_features(set);
    unsigned int l;
    unsigned int L = layers;
    
    fp = fopen(path, "w");
    
    if ( fp == NULL ) 
    {
        printf("%s error: Failed to open file: %s for writing.\n",
               __func__, path);
        exit(1);
    }
    
    /*
     * LSTM net file structure
     * File version   BINARY_FILE_VERSION
     * NbrFeatures    (F)
     * NbrLayers      (L)
     * Inputs  layer  1 (output layer)
     * Nodes   layer  1
     * outputs layer  1
     * Inputs  layer  2
     * Nodes   layer  2
     * outputs layer  2
     * ...
     * Inputs  layer  L (input layer)
     * Nodes   layer  L
     * outputs layer  L
     * Feature Value  1 (int in ASCII [0-255])
     * Feature Value  2
     * ...
     * Feature Value  F
     * --- From here on it is a blob of bytes ---
     * Layer 1: Wy
     * Layer 1: Wi
     * Layer 1: Wc
     * Layer 1: Wo
     * Layer 1: Wf
     * Layer 1: by
     * Layer 1: bi
     * Layer 1: bc
     * Layer 1: bf
     * Layer 1: bo
     * ...
     * Layer L: Wy
     * Layer L: Wi
     * ...
     */
    
    // Write file version
    fprintf(fp, "%d\r\n", BINARY_FILE_VERSION);
    
    // Write NbrFeatures
    fprintf(fp, "%d\r\n", F);
    
    // Write NbrLayers
    fprintf(fp, "%d\r\n", L);
    
    l = 0;
    while ( l < L ) {
        // write number of inputs, nodes and outputs in this layer
        fprintf(fp, "%d\r\n%d\r\n%d\r\n",
                model[l]->X, model[l]->N, model[l]->Y);
        ++l;
    }
    
    // Write feature set
    f = 0;
    while ( f < F )
    {
        fprintf(fp, "%d\r\n", set->values[f]);
        ++f;
    }
    
    // Write the network weights
    lstm_store_net_layers(model, fp, L);
    
    fclose(fp);
}

int lstm_reinit_model(lstm_model_t** model, unsigned int layers,
                      unsigned int previousNbrFeatures, unsigned int newNbrFeatures)
{
    /* Expand last and first layer, add newNbrFeatures - previousNbrFeatures
     *  rows with random initialized weights. */
    lstm_model_t* modelInputs;
    lstm_model_t* modelOutputs;
    
    int Sold = model[layers-1]->S;
    int Snew = newNbrFeatures + model[layers-1]->N;
    int Nin = model[layers-1]->N;
    int Nout;
    int Yold = previousNbrFeatures;
    int Ynew = newNbrFeatures;
    int i, n;
    
    numeric_t *newVectorWf;
    numeric_t *newVectorWi;
    numeric_t *newVectorWc;
    numeric_t *newVectorWo;
    numeric_t *newVectorWy;
    
    /* Sanity checks.. */
    if ( layers == 0 )
        return -1;
    
    /* Sanity checks, don't report back error codes because.. */
    if ( previousNbrFeatures == newNbrFeatures ||
        previousNbrFeatures > newNbrFeatures )
        return -1;
    
    assert(previousNbrFeatures < newNbrFeatures);
    assert(Sold < Snew);
    
    Nout = model[0]->N;
    
    modelOutputs = model[0];
    modelInputs = model[layers-1];
    
    // Reallocate the vectors that depend on input size
    newVectorWf = get_random_vector(Nin * Snew, Snew*5);
    newVectorWi = get_random_vector(Nin * Snew, Snew*5);
    newVectorWc = get_random_vector(Nin * Snew, Snew*5);
    newVectorWo = get_random_vector(Nin * Snew, Snew*5);
    
    n = 0;
    while ( n < Nin )
    {
        i = 0;
        while ( i < Sold )
        {
            newVectorWf[n*Snew + i] = modelInputs->W[LSTM_WB_F][n*Sold + i];
            newVectorWi[n*Snew + i] = modelInputs->W[LSTM_WB_I][n*Sold + i];
            newVectorWc[n*Snew + i] = modelInputs->W[LSTM_WB_C][n*Sold + i];
            newVectorWo[n*Snew + i] = modelInputs->W[LSTM_WB_O][n*Sold + i];
            ++i;
        }
        ++n;
    }
    
    free(modelInputs->W[LSTM_WB_F]);
    free(modelInputs->W[LSTM_WB_I]);
    free(modelInputs->W[LSTM_WB_C]);
    free(modelInputs->W[LSTM_WB_O]);
    free(modelInputs->dldXc);
    free(modelInputs->dldXo);
    free(modelInputs->dldXi);
    free(modelInputs->dldXf);
    free(modelInputs->Wm[LSTM_WB_F]);
    free(modelInputs->Wm[LSTM_WB_I]);
    free(modelInputs->Wm[LSTM_WB_C]);
    free(modelInputs->Wm[LSTM_WB_O]);
    
    modelInputs->W[LSTM_WB_F] = newVectorWf;
    modelInputs->W[LSTM_WB_I] = newVectorWi;
    modelInputs->W[LSTM_WB_C] = newVectorWc;
    modelInputs->W[LSTM_WB_O] = newVectorWo;
    
    modelInputs->dldXc = get_zero_vector(Snew);
    modelInputs->dldXo = get_zero_vector(Snew);
    modelInputs->dldXi = get_zero_vector(Snew);
    modelInputs->dldXf = get_zero_vector(Snew);
    
    modelInputs->Wm[LSTM_WB_F] = get_zero_vector(Nin * Snew);
    modelInputs->Wm[LSTM_WB_I] = get_zero_vector(Nin * Snew);
    modelInputs->Wm[LSTM_WB_C] = get_zero_vector(Nin * Snew);
    modelInputs->Wm[LSTM_WB_O] = get_zero_vector(Nin * Snew);
    
    // Reallocate vectors that depend on output size
    newVectorWy = get_random_vector(Ynew * Nout, Nout);
    n = 0;
    while ( n < Yold )
    {
        i = 0;
        while ( i < Nout )
        {
            newVectorWy[n*Nout + i] = modelOutputs->W[LSTM_WB_Y][n*Nout + i];
            ++i;
        }
        ++n;
    }
    
    free(modelOutputs->W[LSTM_WB_Y]);
    free(modelOutputs->b[LSTM_WB_Y]);
    free(modelOutputs->Wm[LSTM_WB_Y]);
    free(modelOutputs->bm[LSTM_WB_Y]);
    
    modelOutputs->W[LSTM_WB_Y] = newVectorWy;
    modelOutputs->b[LSTM_WB_Y] = get_zero_vector(Ynew);
    modelOutputs->Wm[LSTM_WB_Y] = get_zero_vector(Ynew * Nout);
    modelOutputs->bm[LSTM_WB_Y] = get_zero_vector(Ynew);
    
    // Set new information
    modelInputs->X = newNbrFeatures;
    modelInputs->S = Snew;
    modelOutputs->Y = newNbrFeatures;
    
    return 0;
}

void lstm_read_net_layers(lstm_model_t** model, FILE *fp, unsigned int layers)
{
    // Will only work for ( layer1->N, layer1->F ) == ( layer2->N, layer2->F )
    unsigned int p = 0;
    
    while ( p < layers )
    {
#ifdef STORE_NET_AS_ASCII
        vector_read_ascii(model[p]->W[LSTM_WB_Y], model[p]->Y * model[p]->N, fp);
        vector_read_ascii(model[p]->W[LSTM_WB_I], model[p]->N * model[p]->S, fp);
        vector_read_ascii(model[p]->W[LSTM_WB_C], model[p]->N * model[p]->S, fp);
        vector_read_ascii(model[p]->W[LSTM_WB_O], model[p]->N * model[p]->S, fp);
        vector_read_ascii(model[p]->W[LSTM_WB_F], model[p]->N * model[p]->S, fp);
        
        vector_read_ascii(model[p]->b[LSTM_WB_Y], model[p]->Y, fp);
        vector_read_ascii(model[p]->b[LSTM_WB_I], model[p]->N, fp);
        vector_read_ascii(model[p]->b[LSTM_WB_C], model[p]->N, fp);
        vector_read_ascii(model[p]->b[LSTM_WB_F], model[p]->N, fp);
        vector_read_ascii(model[p]->b[LSTM_WB_O], model[p]->N, fp);
#else
        vector_read(model[p]->W[LSTM_WB_Y], model[p]->Y * model[p]->N, fp);
        vector_read(model[p]->W[LSTM_WB_I], model[p]->N * model[p]->S, fp);
        vector_read(model[p]->W[LSTM_WB_C], model[p]->N * model[p]->S, fp);
        vector_read(model[p]->W[LSTM_WB_O], model[p]->N * model[p]->S, fp);
        vector_read(model[p]->W[LSTM_WB_F], model[p]->N * model[p]->S, fp);
        
        vector_read(model[p]->b[LSTM_WB_Y], model[p]->Y, fp);
        vector_read(model[p]->b[LSTM_WB_I], model[p]->N, fp);
        vector_read(model[p]->b[LSTM_WB_C], model[p]->N, fp);
        vector_read(model[p]->b[LSTM_WB_F], model[p]->N, fp);
        vector_read(model[p]->b[LSTM_WB_O], model[p]->N, fp);
#endif
        
        ++p;
    }
    
}

void lstm_output_string_layers_to_file(FILE * fp,lstm_model_t ** model_layers,
                                       set_t* char_index_mapping, int first,
                                       int numbers_to_display, int layers)
{
    lstm_values_cache_t ***caches_layer;
    int i = 0, count, index, p = 0, b = 0;
    int input = set_indx_to_char(char_index_mapping, first);
    int Y = model_layers[0]->Y;
    int N = model_layers[0]->N;
#ifdef _WIN32
    numeric_t *first_layer_input = NULL;
#else
    numeric_t first_layer_input[Y];
#endif
    
    if ( fp == NULL )
        return;
    
#ifdef _WIN32
    first_layer_input = malloc(Y*sizeof(numeric_t));
    
    if ( first_layer_input == NULL ) 
    {
        fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n",
                __FILE__, __func__, __LINE__, Y*sizeof(numeric_t));
        exit(1);
    }
#endif
    
    caches_layer = e_calloc(layers, sizeof(lstm_values_cache_t**));
    
    p = 0;
    while ( p < layers )
    {
        caches_layer[p] = e_calloc(2, sizeof(lstm_values_cache_t*));
        
        b = 0;
        while ( b < 2 )
        {
            caches_layer[p][b] = lstm_cache_container_init(
                                                           model_layers[p]->X,
                                                           model_layers[p]->N,
                                                           model_layers[p]->Y);
            ++b;
        }
        ++p;
    }
    
    lstm_cache_container_set_start(caches_layer[0][0], N);
    lstm_cache_container_set_start(caches_layer[0][0], N);
    
    while ( i < numbers_to_display )
    {
        index = set_char_to_indx(char_index_mapping,input);
        
        count = 0;
        while ( count < Y )
        {
            first_layer_input[count] = 0.0;
            ++count;
        }
               
        lstm_forward_propagate(layers, first_layer_input, model_layers, caches_layer, i % 2, (i+1)%2);
 
        input = set_probability_choice(char_index_mapping, caches_layer[0][(i+1)%2]->probs);
        fprintf (fp, "%c", input );
        ++i;
    }
    
    p = 0;
    while ( p < layers )
    {
        b = 0;
        while ( b < 2 ) {
            lstm_cache_container_free( caches_layer[p][b]);
            free(caches_layer[p][b]);
            ++b;
        }
        free(caches_layer[p]);
        ++p;
    }
    
    free(caches_layer);
#ifdef _WIN32
    free(first_layer_input);
#endif
}

void lstm_output_string_layers(lstm_model_t ** model_layers, set_t* char_index_mapping,
                               int first, int numbers_to_display, int layers)
{
    lstm_values_cache_t ***caches_layer;
    int i = 0, count, index, p = 0, b = 0;
    int input = set_indx_to_char(char_index_mapping, first);
    int Y = model_layers[0]->Y;
    int N = model_layers[0]->N;
#ifdef _WIN32
    numeric_t * first_layer_input = NULL;
#else
    numeric_t first_layer_input[Y];
#endif
    
#ifdef _WIN32
    first_layer_input = malloc(Y*sizeof(numeric_t));
    if ( first_layer_input == NULL ) 
    {
        fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n",
                __FILE__, __func__, __LINE__, Y*sizeof(numeric_t));
        exit(1);
    }
#endif
    
    caches_layer = e_calloc(layers, sizeof(lstm_values_cache_t**));
    
    p = 0;
    while ( p < layers )
    {
        caches_layer[p] = e_calloc(2, sizeof(lstm_values_cache_t*));
        b = 0;
        while ( b < 2 )
        {
            caches_layer[p][b] = lstm_cache_container_init(model_layers[p]->X, model_layers[p]->N, model_layers[p]->Y);
            ++b;
        }
        ++p;
    }
    
    lstm_cache_container_set_start(caches_layer[0][0], N);
    lstm_cache_container_set_start(caches_layer[0][0], N);
    
    while ( i < numbers_to_display )
    {
        
        index = set_char_to_indx(char_index_mapping,input);
        
        count = 0;
        while ( count < Y )
        {
            first_layer_input[count] = 0.0;
            ++count;
        }
        
        if ( index < 0 ) 
        {
            index = 0;
            printf("%s.%s unexpected input char: '%c', (%d)\r\n", __FILE__, __func__, input, input);
        }
        
        first_layer_input[index] = 1.0;
        
        lstm_forward_propagate(layers, first_layer_input, model_layers, caches_layer, i % 2, (i+1)%2);

        input = set_probability_choice(char_index_mapping, caches_layer[0][(i+1)%2]->probs);
        printf ( "%c", input );
        
        ++i;
    }
    
    p = 0;
    while ( p < layers )
    {
        
        b = 0;
        while ( b < 2 )
        {
            lstm_cache_container_free( caches_layer[p][b]);
            free(caches_layer[p][b]);
            ++b;
        }
        free(caches_layer[p]);
        ++p;
    }
    
    free(caches_layer);
#ifdef _WIN32
    free(first_layer_input);
#endif
}

void lstm_output_string_from_string(lstm_model_t **model_layers, set_t* char_index_mapping,
                                    char * input_string, int layers, int out_length)
{
    lstm_values_cache_t ***caches_layers;
    int i = 0, count, index, in_len;
    char input;
    int Y = model_layers[0]->Y;    
    int p = 0;
    
#ifdef _WIN32
    numeric_t *first_layer_input = malloc(Y*sizeof(numeric_t));
    
    if ( first_layer_input == NULL ) 
    {
        fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n",
                __FILE__, __func__, __LINE__, Y*sizeof(numeric_t));
        exit(1);
    }
    memset(first_layer_input, 0, Y * sizeof(numeric_t));
#else
    numeric_t first_layer_input[Y];
#endif
    
    caches_layers = e_calloc(layers, sizeof(lstm_values_cache_t**));
    
    while ( p < layers )
    {
        caches_layers[p] = e_calloc(2, sizeof(lstm_values_cache_t*));
        i = 0;
        while ( i < 2 )
        {
            caches_layers[p][i] = lstm_cache_container_init(model_layers[p]->X, model_layers[0]->N, model_layers[0]->Y);
            ++i;
        }
        ++p;
    }
    
    in_len = (int)strlen(input_string);
    i = 0;
    
    while ( i < in_len )
    {
        printf("%c", input_string[i]);
        index = set_char_to_indx(char_index_mapping, input_string[i]);
        
        count = 0;
        while ( count < Y )
        {
            first_layer_input[count] = count == index ? 1.0 : 0.0;
            ++count;
        }
        
        lstm_forward_propagate(layers, first_layer_input, model_layers, caches_layers, i % 2, (i+1)%2);
        ++i;
    }
    
    input = set_probability_choice(char_index_mapping, caches_layers[0][i%2]->probs);
    
    printf("%c", input);
    i = 0;
    while ( i < out_length )
    {
        index = set_char_to_indx(char_index_mapping,input);
        
        count = 0;
        while ( count < Y )
        {
            first_layer_input[count] = (count == index) ? 1.0 : 0.0;
            ++count;
        }
        
        lstm_forward_propagate(layers, first_layer_input, model_layers, caches_layers, i % 2, (i+1)%2);

        input = set_probability_choice(char_index_mapping, caches_layers[0][(i+1)%2]->probs);
        printf ( "%c", input );
        //    set_print(char_index_mapping,caches_layer_one->probs);
        ++i;
    }
    
    printf("\n");
    
    p = 0;
    while ( p < layers )
    {        
        i = 0;
        while ( i < 2 ) 
        {
            lstm_cache_container_free( caches_layers[p][i] );
            free(caches_layers[p][i]);
            ++i;
        }
        
        free(caches_layers[p]);        
        ++p;
    }
    
    free(caches_layers);
#ifdef _WIN32
    free(first_layer_input);
#endif
}

void lstm_store_progress(const char* filename, unsigned int n, unsigned int epoch, numeric_t loss, const char * tnh, const char * mt, unsigned int L, unsigned int N)
{
    const net_cpu_performance_t * pcpu = get_cpu_performance();
    FILE * fp = fopen(filename, "a");
    if (NULL != fp)
    {
        fprintf(fp, "%s,%s,%u,%u,%u,%u,%f,%lf,%lf,%lf,%lf,%lf,%lf\n",
                tnh, mt, L, N, n, epoch, loss,
                pcpu->cpu_time_forward, pcpu->time_forward,
                pcpu->cpu_time_back, pcpu->time_back,
                pcpu->cpu_time_adam, pcpu->time_adam);
        fclose(fp);
    }
}

void lstm_model_regularization(lstm_model_t* model, lstm_model_t* gradients)
{
    numeric_t lambda = model->params->lambda;
    
    vectors_add_scalar_multiply(gradients->W[LSTM_WB_Y], model->W[LSTM_WB_Y], model->Y * model->N, lambda);
    vectors_add_scalar_multiply(gradients->W[LSTM_WB_I], model->W[LSTM_WB_I], model->N * model->S, lambda);
    vectors_add_scalar_multiply(gradients->W[LSTM_WB_C], model->W[LSTM_WB_C], model->N * model->S, lambda);
    vectors_add_scalar_multiply(gradients->W[LSTM_WB_O], model->W[LSTM_WB_O], model->N * model->S, lambda);
    vectors_add_scalar_multiply(gradients->W[LSTM_WB_F], model->W[LSTM_WB_F], model->N * model->S, lambda);
    
    vectors_add_scalar_multiply(gradients->b[LSTM_WB_Y], model->b[LSTM_WB_Y], model->Y, lambda);
    vectors_add_scalar_multiply(gradients->b[LSTM_WB_I], model->b[LSTM_WB_I], model->N, lambda);
    vectors_add_scalar_multiply(gradients->b[LSTM_WB_C], model->b[LSTM_WB_C], model->N, lambda);
    vectors_add_scalar_multiply(gradients->b[LSTM_WB_O], model->b[LSTM_WB_O], model->N, lambda);
    vectors_add_scalar_multiply(gradients->b[LSTM_WB_F], model->b[LSTM_WB_F], model->N, lambda);
}

//                        model, number of training points, X_train, Y_train
void lstm_train(lstm_model_t** model_layers, lstm_model_parameters_t *params,
                set_t* char_index_mapping, unsigned int training_points,
                int* X_train, int* Y_train, unsigned int layers, numeric_t *loss_out)
{
    unsigned int p, i = 0, b = 0, q = 0, e1 = 0, e2 = 0,
    e3 = 0, record_iteration = 0, tmp_count, trailing;
    unsigned int n = 0, epoch = 0;
    numeric_t loss = -1, loss_tmp = 0.0, record_keeper = 0.0;
    numeric_t initial_learning_rate = params->learning_rate;
    time_t time_iter;
    char time_buffer[40];
    unsigned long iterations = params->iterations;
    unsigned long epochs = params->epochs;
    int stateful = params->stateful, decrease_lr = params->decrease_lr;
    // configuration for output printing during training
    int print_progress = params->print_progress;
    long print_progress_iterations = params->print_progress_iterations;
    int print_progress_sample_output = params->print_progress_sample_output;
    int print_progress_to_file = params->print_progress_to_file;
    int print_progress_number_of_chars = params->print_progress_number_of_chars;
    char *print_progress_to_file_name = params->print_sample_output_to_file_name;
    char *print_progress_to_file_arg = params->print_sample_output_to_file_arg;
    int store_progress_every_x_iterations = params->store_progress_every_x_iterations;
    char *store_progress_file_name = params->store_progress_file_name;
    int store_network_every = params->store_network_every;

    lstm_values_state_t ** stateful_d_next = NULL;
    lstm_values_cache_t ***cache_layers;
    lstm_values_next_cache_t **d_next_layers;
    
    lstm_model_t **gradient_layers, **gradient_layers_entry,  **M_layers = NULL, **R_layers = NULL;
    
#ifdef _WIN32
    numeric_t *first_layer_input = malloc(model_layers[0]->Y*sizeof(numeric_t));
    
    if ( first_layer_input == NULL )
    {
        fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n",
                __FILE__, __func__, __LINE__, model_layers[0]->Y*sizeof(numeric_t));
        exit(1);
    }
#else
    numeric_t first_layer_input[model_layers[0]->Y];
#endif
    
    if ( stateful )
    {
        stateful_d_next = e_calloc(layers, sizeof(lstm_values_state_t*));
        
        i = 0;
        while ( i < layers )
        {
            stateful_d_next[i] = e_calloc( training_points/params->mini_batch_size + 1, sizeof(lstm_values_state_t));
            
            lstm_values_state_init(&stateful_d_next[i], model_layers[i]->N);
            ++i;
        }
    }
    
    i = 0;
    cache_layers = e_calloc(layers, sizeof(lstm_values_cache_t**));
    
    while ( i < layers )
    {
        cache_layers[i] = e_calloc(params->mini_batch_size + 1,
                                   sizeof(lstm_values_cache_t*));
        
        p = 0;
        while ( p < params->mini_batch_size + 1 )
        {
            cache_layers[i][p] = lstm_cache_container_init(model_layers[i]->X, model_layers[i]->N, model_layers[i]->Y);
            if ( cache_layers[i][p] == NULL )
                lstm_init_fail("Failed to allocate memory for the caches\n");
            ++p;
        }
        
        ++i;
    }
    
    gradient_layers = e_calloc(layers, sizeof(lstm_model_t*) );
    
    gradient_layers_entry = e_calloc(layers, sizeof(lstm_model_t*) );
    
    d_next_layers = e_calloc(layers, sizeof(lstm_values_next_cache_t *));
    
    if ( params->optimizer == OPTIMIZE_ADAM ) {
        
        M_layers = e_calloc(layers, sizeof(lstm_model_t*) );
        R_layers = e_calloc(layers, sizeof(lstm_model_t*) );
        
    }
    
    i = 0;
    while ( i < layers )
    {
        lstm_init_model(model_layers[i]->X,
                        model_layers[i]->N, model_layers[i]->Y,
                        &gradient_layers[i], 1, params);
        lstm_init_model(model_layers[i]->X,
                        model_layers[i]->N, model_layers[i]->Y,
                        &gradient_layers_entry[i], 1, params);
        lstm_values_next_cache_init(&d_next_layers[i],
                                    model_layers[i]->N, model_layers[i]->X);
        
        if ( params->optimizer == OPTIMIZE_ADAM )
        {
            lstm_init_model(model_layers[i]->X,
                            model_layers[i]->N, model_layers[i]->Y, &M_layers[i], 1, params);
            lstm_init_model(model_layers[i]->X,
                            model_layers[i]->N, model_layers[i]->Y, &R_layers[i], 1, params);
        }
        
        ++i;
    }
    
    
    i = 0; b = 0;
    while ( n < iterations )
    {
        
        if ( epochs && epoch >= epochs )
        {
            // We have done enough iterations now
            break;
        }
        
        b = i;
        
        loss_tmp = 0.0;
        
        q = 0;
        
        while (q < layers) 
        {
            if (stateful)
            {
                if ( q == 0 )
                    lstm_cache_container_set_start(cache_layers[q][0],  model_layers[q]->N);
                else
                    lstm_next_state_copy(stateful_d_next[q], cache_layers[q][0], model_layers[q]->N, 0);
            }
            else
            {
                lstm_cache_container_set_start(cache_layers[q][0], model_layers[q]->N);
            }
            ++q;
        }
        
        unsigned int check = i % training_points;
        trailing = params->mini_batch_size;
        
        if ( i + params->mini_batch_size >= training_points )
        {
            trailing = training_points - i;
        }
        
        q = 0;
        
        // forward propagate
        while ( q < trailing )
        {
            e1 = q;
            e2 = q + 1;
            
            e3 = i % training_points;
            
            tmp_count = 0;
            while ( tmp_count < model_layers[0]->Y )
            {
                first_layer_input[tmp_count] = 0.0;
                ++tmp_count;
            }
            
            first_layer_input[X_train[e3]] = 1.0;
            
            lstm_forward_propagate(layers, first_layer_input, model_layers, cache_layers, e1, e2);

            loss_tmp += cross_entropy(cache_layers[0][e2]->probs, Y_train[e3]);
            ++i; ++q;
        }
        
        loss_tmp /= (q+1);
        
        if ( loss < 0 )
            loss = loss_tmp;
        
        loss = loss_tmp * params->loss_moving_avg + (1 - params->loss_moving_avg) * loss;
        
        if ( n == 0 )
            record_keeper = loss;
        
        if ( loss < record_keeper )
        {
            record_keeper = loss;
            record_iteration = n;
        }
        
        if ( stateful )
        {
            p = 0;
            while ( p < layers )
            {
                lstm_next_state_copy(stateful_d_next[p], cache_layers[p][e2], model_layers[p]->N, 1);
                ++p;
            }
        }
        
        p = 0;
        while ( p < layers )
        {
            lstm_zero_the_model(gradient_layers[p]);
            lstm_zero_d_next(d_next_layers[p], model_layers[p]->X, model_layers[p]->N);
            ++p;
        }
         
        // Back propagate
        while ( q > 0 )
        {
            e1 = q;
            e2 = q - 1;
            
            e3 = ( training_points + i - 1 ) % training_points;
            
            p = 0;
            while ( p < layers )
            {
                lstm_zero_the_model(gradient_layers_entry[p]);
                ++p;
            }
                        
            lstm_backward_propagate(layers,
                                    model_layers,
                                    cache_layers,
                                    Y_train,
                                    d_next_layers,
                                    gradient_layers_entry,
                                    e1, e3);
            p = 0;
            while ( p < layers ) 
            {
                sum_gradients(gradient_layers[p], gradient_layers_entry[p]);
                ++p;
            }
            
            i--; q--;
        }
        
        assert(check == e3);
        
        for (p = 0; p < layers; p++)
        {            
            if ( params->gradient_clip )
                gradients_clip(gradient_layers[p], params->gradient_clip_limit);
            
            if ( params->gradient_fit )
                gradients_fit(gradient_layers[p], params->gradient_clip_limit);
        }
        p = 0;
        
        switch ( params->optimizer )
        {
            case OPTIMIZE_ADAM:
                adam_optimize(layers, model_layers, gradient_layers, M_layers, R_layers, n, params->use_threads);
                break;

            case OPTIMIZE_GRADIENT_DESCENT:
                while ( p < layers ) 
                {
                    gradients_decend(model_layers[p], gradient_layers[p]);
                    ++p;
                }
                break;

            default:
                fprintf( stderr,
                        "Failed to update gradients, no acceptable optimization algorithm provided.\n"
                        "lstm_model_parameters_t has a field called 'optimizer'. Set this value to:\n"
                        "%d: Adam gradients optimizer algorithm\n"
                        "%d: Gradients descent algorithm.\n",
                        OPTIMIZE_ADAM,
                        OPTIMIZE_GRADIENT_DESCENT
                        );
                exit(1);
                break;
        }
        
        if ( print_progress && !( n % print_progress_iterations ) )
        {
            memset(time_buffer, '\0', sizeof time_buffer);
            time(&time_iter);
            strftime(time_buffer, sizeof time_buffer, "%X", localtime(&time_iter));
            const net_cpu_performance_t * pcpu = get_cpu_performance();
            
            printf("%s Iteration: %u (epoch: %u), Loss: %f, record: %f (iteration: %d), LR: %f\n",
                   time_buffer, n, epoch, loss, record_keeper, record_iteration, params->learning_rate);
            printf("Using %s: Total backward time: %.3f  Total forward time: %.3f  Total optimizer time: %.3f\n",
                   params->use_tanf != 0 ? "TANF" : "TANH", pcpu->time_back, pcpu->time_forward, pcpu->time_adam);
            
            if ( print_progress_sample_output )
            {
                printf("=====================================================\n");
                lstm_output_string_layers(model_layers, char_index_mapping, X_train[b],
                                          print_progress_number_of_chars, layers);
                printf("\n=====================================================\n");
            }
            
            if ( print_progress_to_file )
            {
                FILE * fp_progress_output = fopen(print_progress_to_file_name,
                                                  print_progress_to_file_arg);
                if ( fp_progress_output != NULL )
                {
                    fprintf(fp_progress_output, "%s====== Iteration: %u, loss: %.5lf ======\n", n==0 ? "" : "\n", n, loss);
                    printf("==== %s: Backward time: %.3f.  Forward time: %.3f.  Optimizer time: %.3f  ======\n",
                           params->use_tanf != 0 ? "TANF" : "TANH",
                           pcpu->time_back, pcpu->time_forward, pcpu->time_adam);
                    lstm_output_string_layers_to_file(fp_progress_output, model_layers, char_index_mapping, X_train[b], print_progress_number_of_chars, layers);
                    fclose(fp_progress_output);
                }
            }
            
            // Flushing stdout
            fflush(stdout);
        }
        
        if ( store_progress_every_x_iterations && !(n % store_progress_every_x_iterations ))
        {
            const char * mt = params->use_threads ? "Multi-threaded" : "Single-threaded";
            const char* th = params->use_tanf ? "tanf" : "tanh";
            lstm_store_progress(store_progress_file_name, n, epoch, loss, th, mt, params->layers, params->neurons);
        }
        
        if ( store_network_every && !(n % store_network_every) )
        {
            lstm_store(
                       params->store_network_name_raw,
                       char_index_mapping,
                       model_layers,
                       layers);
            lstm_store_net_layers_as_json(model_layers, params->store_network_name_json,
                                          params->store_char_indx_map_name, char_index_mapping, layers);
        }
        
        if ( b + params->mini_batch_size >= training_points )
            epoch++;
        
        i = (b + params->mini_batch_size) % training_points;
        
        if ( i < params->mini_batch_size )
            i = 0;
        
        if ( decrease_lr ) 
        {
            params->learning_rate = initial_learning_rate / ( 1.0 + n / params->learning_rate_decrease );
            //printf("learning rate: %f\n", model->params->learning_rate);
        }        
        ++n;
    }
    
    // Reporting the loss value
    *loss_out = loss;
    
    p = 0;
    while ( p < layers ) 
    {
        lstm_values_next_cache_free(d_next_layers[p]);
        
        i = 0;
        while ( i < params->mini_batch_size ) 
        {
            lstm_cache_container_free(cache_layers[p][i]);
            lstm_cache_container_free(cache_layers[p][i]);
            ++i;
        }
        
        if ( params->optimizer == OPTIMIZE_ADAM ) 
        {
            lstm_free_model(M_layers[p]);
            lstm_free_model(R_layers[p]);
        }
        
        lstm_free_model(gradient_layers_entry[p]);
        lstm_free_model(gradient_layers[p]);
        
        ++p;
    }
    
    if ( stateful && stateful_d_next != NULL ) 
    {
        for (i = 0; i < layers; i++)
        {
            free(stateful_d_next[i]);
        }
        free(stateful_d_next);
    }
    
    
    free(cache_layers);
    free(gradient_layers);
    if ( M_layers != NULL )
        free(M_layers);
    if ( R_layers != NULL )
        free(R_layers);
#ifdef _WIN32
    free(first_layer_input);
#endif
}

