/*
 * This file is part of the LSTM Network implementation In C made by Rickard Hallerbäck
 *
 *                 Copyright (c) 2018 Rickard Hallerbäck
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
 * Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies
 * or substantial portions of the Software.
 *
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 */

/*! \file lstm.h
 \brief LSTM functionalities [Perhaps most interesting from a user perspective]
 
 Here are some functions that define, produce and use the LSTM network.
 
 A LSTM network consists of an array of \ref lstm_model_t data.
 
 Each model in this array is called a layer. Output is collected
 in the beginning of the array, and input is entered in the end.
 
 Say there is L > 0 layers, in a model defined as \ref lstm_model_t ** model,
 then model[0] refers to the output layer and model[L-1] refers to the input layer.
 */

#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>

#ifdef _WIN32

#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "utilities.h"
#include "set.h"
#include "layers.h"
#include "assert.h"

#define OPTIMIZE_ADAM                       0
#define OPTIMIZE_GRADIENT_DESCENT           1

#define LSTM_MAX_LAYERS                     10
#define LSTM_MAX_NEURONS                    300

#define BINARY_FILE_VERSION                 1

#define EPSILON                             1e-7

typedef struct lstm_model_parameters_t
{
    // For progress monitoring
    numeric_t loss_moving_avg;
    // For gradient descent
    numeric_t learning_rate;
    numeric_t momentum;
    numeric_t lambda;
    numeric_t softmax_temp;
    numeric_t beta1;
    numeric_t beta2;
    int gradient_clip;
    int gradient_fit;
    int optimizer;
    int model_regularize;
    int stateful;
    int decrease_lr;
    numeric_t learning_rate_decrease;
    
    // How many layers
    unsigned int layers;
    // How many neurons this layer has
    unsigned int neurons;
    
    // Output configuration for interactivity
    long print_progress_iterations;
    int  print_progress_sample_output;
    int  print_progress;
    int  print_progress_to_file;
    int  print_progress_number_of_chars;
    char *print_sample_output_to_file_name;
    char *print_sample_output_to_file_arg;
    int  store_progress_every_x_iterations;
    char *store_progress_file_name;
    int  store_network_every;
    char *store_network_name_raw;
    char *store_network_name_json;
    char *store_char_indx_map_name;
    
    // General parameters
    unsigned int mini_batch_size;
    numeric_t gradient_clip_limit;
    unsigned long iterations;
    unsigned long epochs;
    
    int use_tanf;           // use tanf()
    int use_threads;        // use multiple threads
} lstm_model_parameters_t;

typedef enum  lstm_model_wb
{
    LSTM_WB_F = 0,
    LSTM_WB_I,
    LSTM_WB_C,
    LSTM_WB_O,
    LSTM_WB_Y,
    LSTM_PARAMTERS
} lstm_model_wb_t;

typedef struct lstm_model_t
{
    unsigned int X; /**< Number of input nodes */
    unsigned int N; /**< Number of neurons */
    unsigned int Y; /**< Number of output nodes */
    unsigned int S; /**< lstm_model_t.X + lstm_model_t.N */
    
    // Parameters
    lstm_model_parameters_t * params;
    
    // The model
    numeric_t* W[LSTM_PARAMTERS];
    numeric_t* b[LSTM_PARAMTERS];

    numeric_t* Wm[LSTM_PARAMTERS];
    numeric_t* bm[LSTM_PARAMTERS];

//    numeric_t* Wf;
//    numeric_t* Wi;
//    numeric_t* Wc;
//    numeric_t* Wo;
//    numeric_t* Wy;
//
//    numeric_t* bf;
//    numeric_t* bi;
//    numeric_t* bc;
//    numeric_t* bo;
//    numeric_t* by;
    
    // cache
    numeric_t* dldh;
    numeric_t* dldho;
    numeric_t* dldhf;
    numeric_t* dldhi;
    numeric_t* dldhc;
    numeric_t* dldc;
    
    numeric_t* dldXi;
    numeric_t* dldXo;
    numeric_t* dldXf;
    numeric_t* dldXc;
    
    // Gradient descent momentum
//    numeric_t* Wfm;
//    numeric_t* Wim;
//    numeric_t* Wcm;
//    numeric_t* Wom;
//    numeric_t* Wym;
//
//    numeric_t* bfm;
//    numeric_t* bim;
//    numeric_t* bcm;
//    numeric_t* bom;
//    numeric_t* bym;
        
} lstm_model_t;

typedef struct lstm_values_cache_t {
    numeric_t* probs;
    numeric_t* probs_before_sigma;
    numeric_t* c;
    numeric_t* h;
    numeric_t* c_old;
    numeric_t* h_old;
    numeric_t* X;
    numeric_t* hf;
    numeric_t* hi;
    numeric_t* ho;
    numeric_t* hc;
    numeric_t* tanh_c_cache;
} lstm_values_cache_t;

typedef struct lstm_values_state_t {
    numeric_t* c;
    numeric_t* h;
} lstm_values_state_t;

typedef struct lstm_values_next_cache_t {
    numeric_t* dldh_next;
    numeric_t* dldc_next;
    numeric_t* dldY_pass;
} lstm_values_next_cache_t;

/**
 * Initialize a new model
 * @param X number of inputs
 * @param N number of nodes
 * @param Y number of outputs
 * @param model_to_be_set the model object that is to be set
 * @param zeros if set to 0, the model will have zeros as weights,\
 otherwise random initialization
 * @param params model parameters
 * @return 0 on success, negative values on errors
 */
int lstm_init_model(int X, int N, int Y,
                    lstm_model_t** model_to_be_set, int zeros,
                    lstm_model_parameters_t *params);
/**
 * Set all weights in a model to zero
 * @param model model to be set to zero
 */
void lstm_zero_the_model(lstm_model_t *model);
void lstm_zero_d_next(lstm_values_next_cache_t * d_next,
                      int outputs, int neurons);
void lstm_cache_container_set_start(lstm_values_cache_t *cache, int neurons);

/**
 * Free a model
 * @param lstm model to be freed
 */
void lstm_free_model(lstm_model_t *lstm);
/**
 * Compute the output of a network
 * @param model model to be used, must been initialized with \ref lstm_init_model
 * \see lstm_init_model
 */
void lstm_forward_propagate(lstm_model_t *model, numeric_t *input,
                            lstm_values_cache_t *cache_in, lstm_values_cache_t *cache_out, int softmax);

void lstm_values_state_init(lstm_values_state_t** d_next_to_set, int N);
void lstm_values_next_state_free(lstm_values_state_t* d_next);

lstm_values_cache_t*  lstm_cache_container_init(int X, int N, int Y);
void lstm_cache_container_free(lstm_values_cache_t*);
void lstm_values_next_cache_init(lstm_values_next_cache_t**, int N, int X);
void lstm_values_next_cache_free(lstm_values_next_cache_t*);
void sum_gradients(lstm_model_t*, lstm_model_t*);

/**
 * Load a previously stored network, generated with \ref lstm_store
 * \see lstm_store
 * @param path path to the model that is to be loaded
 * @param set feature set, will be read from network
 * @param params parameters, some will be read from loaded
 * @param model model reference to be set
 */
void lstm_load(const char *path, set_t *set,
               lstm_model_parameters_t *params, lstm_model_t ***model);
/**
 * Store a network, can be read again with \ref lstm_load
 * \see lstm_load
 * @param path path to the model that is to be store
 * @param set feature set, will be stored to the file
 * @param model model reference to be stored
 * @param layers number of layers in the model.\
 \p model is an array, layers is used as a length.
 */
void lstm_store(const char *path, set_t *set,
                lstm_model_t **model, unsigned int layers);
int lstm_reinit_model(
                      lstm_model_t** model, unsigned int layers,
                      unsigned int previousNbrFeatures, unsigned int newNbrFeatures);
/**
 * Store a network in JSON format.
 * \see lstm_init
 * \see lstm_train
 * @param model model to be stored
 * @param filename eile that is to be store
 * @param set_name Name of the field that will present the feature-to-index \
 mapping in the JSON file.
 * @param set feature-to-index set, will be stored to the file
 * @param layers the number of layers this model consists of. \
 \p model is an array, layers is used as a length.
 */
void lstm_store_net_layers_as_json(lstm_model_t** model, const char * filename,
                                   const char *set_name, set_t *set, unsigned int layers);
void lstm_store_progress(const char*, unsigned int, unsigned int, numeric_t, const char *, const char *, unsigned int, unsigned int);

/**
 * This is the entry point to the realm of black magic.
 * Trains the network.
 * \see lstm_init_model
 * @param model The model that is to be used, must have been \
 initialzed with \ref lstm_init_model.
 * @param params Various parameters determining the training process
 * @param set The feature-to-index mapping.
 * @param training_points length of the training data array \p X
 * @param X input observations
 * @param Y output observations (typically &X[1], so that X[0] -> Y[0]: 'h' -> 'e', \
 if X[...] = 'hello' => Y[...] = 'ello ').
 * @param layers number of layers in the network, the number of models \p model \
 is pointing to. Internally if layers is L, then input is given to model[L-1] and \
 output collected at model[0].
 * @param loss the value of the loss function, put under a smoothing \
 moving average filter, after the training has been completed.
 */
void lstm_train(lstm_model_t** model, lstm_model_parameters_t*params,
                set_t* set, unsigned int training_points, int *X, int *Y, unsigned int layers,
                numeric_t *loss);
/**
 * If you are training on textual data, this function can be used
 * to sample and output from the network directly to stdout.
 * \see lstm_init_model
 * \see lstm_train
 * @param model_layers The model that is to be used, must have been \
 initialzed with \ref lstm_init_model.
 * @param set The feature-to-index mapping.
 * @param first input seed, the rest will "follow" to stdout.
 * @param samples_to_display How many observations to write to stdout
 * @param layers how many layers this network has
 */
void lstm_output_string_layers(lstm_model_t ** model_layers, set_t* set,
                               int first, int samples_to_display, int layers);
/**
 * If you are training on textual data, this function can be used
 * to sample and output from the network directly to stdout.
 * \see lstm_init_model
 * \see lstm_train
 * @param model The model that is to be used, must have been \
 initialzed with \ref lstm_init_model.
 * @param set The feature-to-index mapping.
 * @param input_string input seed string, the rest will "follow" to stdout.
 * @param layers how many layers this network has
 * @param out_length How many characters to write to stdout
 */
void lstm_output_string_from_string(lstm_model_t **model,
                                    set_t* set, char * input_string, int layers, int out_length);
/**
 * If you are training on textual data, this function can be used
 * to sample and output from the network directly to file.
 * \see lstm_init_model
 * \see lstm_train
 * @param fp an open file handle to which one will write. Must \
 have been opened with write privileges
 * @param model_layers The model that is to be used, must have been \
 initialzed with \ref lstm_init_model.
 * @param set The feature-to-index mapping.
 * @param first input seed, the rest will "follow" to file.
 * @param samples_to_display How many observations to write to stdout
 * @param layers how many layers this network has
 */
void lstm_output_string_layers_to_file(FILE * fp,lstm_model_t ** model_layers,
                                       set_t* set, int first, int samples_to_display, int layers);

void lstm_read_net_layers(lstm_model_t** model, FILE *fp, unsigned int layers);
void lstm_store_net_layers(lstm_model_t** model, FILE *fp, unsigned int layers);

void gradients_adam_optimizer(lstm_model_t* model, lstm_model_t* gradients, lstm_model_t* M, lstm_model_t* R, unsigned int t);

#endif

