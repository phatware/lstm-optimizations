//
//  set.h
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/10/22.
//

#ifndef LSTM_SET_H
#define LSTM_SET_H

/*! \file set.h
    \brief LSTM feature-to-index mapping
    
    Features get mapped to an index value.
    This process is done using the following definitions and functions.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include "utilities.h"

#define	SET_MAX_CHARS	1000

typedef struct set_t {
    char values[SET_MAX_CHARS];
    int free[SET_MAX_CHARS];
} set_t;

int set_insert_symbol(set_t*, char);
char set_indx_to_char(set_t*, int);
int set_char_to_indx(set_t*, char);
int set_probability_choice(set_t*, numeric_t*);
int set_greedy_argmax(set_t*, numeric_t*);
int set_get_features(set_t*);

void set_print(set_t*, numeric_t*);

void initialize_set(set_t*);

void set_store_as_json(set_t *, FILE*);
void set_store(set_t *, FILE*);
int set_read(set_t *, FILE*);

#endif
