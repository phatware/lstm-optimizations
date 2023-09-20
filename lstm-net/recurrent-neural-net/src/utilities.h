//
//  lstm_threaded.hpp
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/10/22.
//

#ifndef LSTM_UTILITIES_H
#define LSTM_UTILITIES_H

/*! \file utilities.h
    \brief Some utility functions used in the LSTM program
    
    Here are some functions that help produce the LSTM network.
*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

#ifdef _SINGLE_PRECISION
typedef float numeric_t;
#else
typedef double numeric_t;
#endif // _SINGLE_PRECISION

#define _randf ((numeric_t) ((double)rand() / (double)RAND_MAX))

// used on contigious vectors
//		A = A + B		A,		B,    l
void 	vectors_add(numeric_t*, numeric_t*, int);
void 	vectors_subtract(numeric_t*, numeric_t*, int);
void 	vectors_add_scalar_multiply(numeric_t*, numeric_t*, int, numeric_t);
void 	vectors_subtract_scalar_multiply(numeric_t*, numeric_t*, int, numeric_t);
void 	vectors_add_scalar(numeric_t*, numeric_t, int );
void 	vectors_div(numeric_t*, numeric_t*, int);
void 	vector_sqrt(numeric_t*, int);
void 	vector_store_json(numeric_t*, int, FILE *);
void 	vector_store_as_matrix_json(numeric_t*, int, int, FILE *);

// A = B * s
void  vectors_copy_multiply_scalar(numeric_t* A, const numeric_t* B, numeric_t s, int L);
// C = A * B
void  vectors_copy_multiply(numeric_t* C, const numeric_t* A, const numeric_t* B,  int L);

//		A = A + B		A,		B,    R, C
void 	matrix_add(numeric_t**, numeric_t**, int, int);
void 	matrix_subtract(numeric_t**, numeric_t**, int, int);
//		A = A*b		A,		b,    R, C
void 	matrix_scalar_multiply(numeric_t**, numeric_t, int, int);

//		A = A * B		A,		B,    l
void 	vectors_multiply(numeric_t*, numeric_t*, int);
//		A = A * b		A,		b,    l
void 	vectors_multiply_scalar(numeric_t*, numeric_t, int);
//		A = random( (R, C) ) / sqrt(R / 2), &A, R, C
int 	init_random_matrix(numeric_t***, int, int);
//		A = 0.0s, &A, R, C
int 	init_zero_matrix(numeric_t***, int, int);
int 	free_matrix(numeric_t**, int);
//						 V to be set, Length

int 	init_zero_vector(numeric_t**, int);
int 	free_vector(numeric_t**);
//		A = B       A,		B,		length
void 	copy_vector(numeric_t*, numeric_t*, int);

numeric_t* 	get_zero_vector(int);
numeric_t** get_zero_matrix(int, int);
numeric_t** get_random_matrix(int, int);
numeric_t* 	get_random_vector(int,int);

void 	matrix_set_to_zero(numeric_t**, int, int);
void 	vector_set_to_zero(numeric_t*, int);

numeric_t sample_normal(void);
numeric_t randn(numeric_t, numeric_t);

numeric_t one_norm(numeric_t*, int);

void matrix_clip(numeric_t**, numeric_t, int, int);
int vectors_fit(numeric_t*, numeric_t, int);
int vectors_clip(numeric_t*, numeric_t, int);

// I/O
void 	vector_print_min_max(char *, numeric_t *, int);
void 	vector_read(numeric_t *, int, FILE *);
void 	vector_store(numeric_t *, int, FILE *);
void 	matrix_store(numeric_t **, int, int, FILE *);  
void 	matrix_read(numeric_t **, int, int, FILE *);
void 	vector_read_ascii(numeric_t *, int, FILE *);
void 	vector_store_ascii(numeric_t *, int, FILE *);


// Memory
void*   e_calloc(size_t count, size_t size);
size_t  e_alloc_total(void);

#endif
