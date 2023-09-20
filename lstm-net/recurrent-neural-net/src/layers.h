//
//  layers.h
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/11/22.
//

#pragma once

#include <stdlib.h>
#include <math.h>
#include "utilities.h"

/*! \file layers.h
 \brief Various mathematical functions
 
 An LSTM network is built out of a series
 of these type of mathematical operations.
 */

// Dealing with FC layers, forward and backward
/**    Y = AX + b
 *
 *  A(rows: R, columns: C)
 */
void fully_connected_forward(numeric_t* Y, const numeric_t* A, const numeric_t* X,
                             const numeric_t* b, int R, int C);
/**        Y = AX + b
 *
 * A(rows: R, columns: C)
 *
 * dld* points to gradients
 */
void fully_connected_backward(const numeric_t* dldY, const numeric_t* A, const numeric_t* X,numeric_t* dldA,
                              numeric_t* dldX, numeric_t* dldb, int R, int C);

/** Softmax layer forward propagation
 *
 * @param P sum ( exp(y / \pa temperature) ) for y in \pa Y
 * @param Y input
 * @param temperature calibration of softmax, the lower the spikier
 * @param F len ( Y )
 */
void softmax_layers_forward(numeric_t* P, const numeric_t* Y, int F, numeric_t temperature);
/** Softmax layer backward propagation
 *
 * @param P sum ( exp(y/temperature) ) for y in Y
 * @param c correct prediction
 * @param dldh gradients back to Y, given \p c
 * @param F len ( Y )
 */
void softmax_loss_layer_backward(const numeric_t* P, int c, numeric_t* dldh, int F);

// Other layers used: sigmoid and tanh
//
/** Y = sigmoid(X)
 *
 * L = len(X)
 */
void sigmoid_forward(numeric_t* Y, const numeric_t* X, int L);
/** Y = sigmoid(X), dldY, Y, &dldX, length */
void sigmoid_backward(const numeric_t* dldY, const numeric_t* Y, numeric_t* dldX, int L);
/** Y = tanh(X), &Y, X, length */
void tanh_forward(numeric_t* Y, const numeric_t* X, int L);
/** Y = tanh(X), dldY, Y, &dldX, length */
void tanh_backward(const numeric_t* dldY, const numeric_t* Y, numeric_t* dldX, int L);
/** Y = tanf(X), dldY, Y, &dldX, length */
void  tanf_forward(numeric_t* Y, const numeric_t* X, int L);
/** Y = tanhf(X), dldY, Y, &dldX, length */
// void  tanf_backward(numeric_t* dldY, numeric_t* Y, numeric_t* dldX, int L, int td);
void  tanf_backward(const numeric_t* dldY, const numeric_t* Y, numeric_t* dldX, int L);


/** The loss function used in the output layer of the LSTM network, which is a softmax layer
 * \see softmax_layers_forward
 * @param probabilities array with output from \ref softmax_layers_forward
 * @param correct the index that represents the correct observation
 */
numeric_t cross_entropy(const numeric_t* probabilities, int correct);


