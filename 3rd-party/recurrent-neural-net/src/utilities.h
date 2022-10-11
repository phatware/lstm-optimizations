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

// used on contigous vectors
//		A = A + B		A,		B,    l
void 	vectors_add(numeric_t*, numeric_t*, int);
void 	vectors_substract(numeric_t*, numeric_t*, int);
void 	vectors_add_scalar_multiply(numeric_t*, numeric_t*, int, numeric_t);
void 	vectors_scalar_multiply(numeric_t*, numeric_t, int);
void 	vectors_substract_scalar_multiply(numeric_t*, numeric_t*, int, numeric_t);
void 	vectors_add_scalar(numeric_t*, numeric_t, int );
void 	vectors_div(numeric_t*, numeric_t*, int);
void 	vector_sqrt(numeric_t*, int);
void 	vector_store_json(numeric_t*, int, FILE *);
void 	vector_store_as_matrix_json(numeric_t*, int, int, FILE *);

//		A = A + B		A,		B,    R, C
void 	matrix_add(numeric_t**, numeric_t**, int, int);
void 	matrix_substract(numeric_t**, numeric_t**, int, int);
//		A = A*b		A,		b,    R, C
void 	matrix_scalar_multiply(numeric_t**, numeric_t, int, int);

//		A = A * B		A,		B,    l
void 	vectors_multiply(numeric_t*, numeric_t*, int);
//		A = A * b		A,		b,    l
void 	vectors_mutliply_scalar(numeric_t*, numeric_t, int);
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
