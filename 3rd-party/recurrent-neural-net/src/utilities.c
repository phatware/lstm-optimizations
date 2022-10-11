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

/*
 * Dealing with common vector operations
 *
 * ==================== WARNING ====================
 *   The caller should have thought about the memory
 *    allocation, these functions assumes that
 *    everything is OK. If not used with care,
 *   prohibted reads/writes might occur.
 * =================================================
 *
 */
#include "utilities.h"
#include <memory.h>

#ifdef _BUILD_FOR_CUDA
#include "cuda_utils.cuh"

#define SHORT_VECTOR    1000
#endif // _BUILD_FOR_CUDA

// used on contigous vectors
void  vectors_add(numeric_t* A, numeric_t* B, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, B, L, 0, vector_math_add))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] += B[l];
        ++l;
    }
}

void  vectors_add_scalar(numeric_t* A, numeric_t B, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, NULL, L, B, vector_math_scalar_add))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] += B;
        ++l;
    }
}

void  vectors_scalar_multiply(numeric_t* A, numeric_t d, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, NULL, L, d, vector_math_scalar_multiply))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] *= d;
        ++l;
    }
}

// A = A + (B * s)
void  vectors_add_scalar_multiply(numeric_t* A, numeric_t* B, int L, numeric_t s)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, B, L, s, vector_math_add_scalar_multiply))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] += B[l] * s;
        ++l;
    }
}

void  vectors_substract(numeric_t* A, numeric_t* B, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, B, L, 0, vector_math_substract))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] -= B[l];
        ++l;
    }
}

void  vectors_div(numeric_t* A, numeric_t* B, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, B, L, 0, vector_math_divide))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] /= B[l];
        ++l;
    }
}

void  vector_sqrt(numeric_t* A, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, NULL, L, 0, vector_math_sqrt))
            return;
    }
#endif
    int l = 0;
    while (l < L) {
        A[l] = sqrt(A[l]);
        ++l;
    }
}

// A = A - (B * s)
void  vectors_substract_scalar_multiply(numeric_t* A, numeric_t* B, int L, numeric_t s)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, B, L, s, vector_math_substact_scalar_multiply))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] -= B[l] * s;
        ++l;
    }
}

void  vectors_multiply(numeric_t* A, numeric_t* B, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, B, L, 0, vector_math_multiply))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] *= B[l];
        ++l;
    }
}

void  vectors_mutliply_scalar(numeric_t* A, numeric_t b, int L)
{
#ifdef _BUILD_FOR_CUDA
    if (L > SHORT_VECTOR)
    {
        if (cudaSuccess == cuda_inplace_vectors_math_op(A, NULL, L, b, vector_math_scalar_multiply))
            return;
    }
#endif // _BUILD_FOR_CUDA
    int l = 0;
    while (l < L) {
        A[l] *= b;
        ++l;
    }
}

int   init_random_matrix(numeric_t*** A, int R, int C)
{
    int r = 0, c = 0;
    
    *A = e_calloc(R, sizeof(numeric_t*));
    
    while ( r < R ) {
        (*A)[r] = e_calloc(C, sizeof(numeric_t));
        ++r;
    }
    
    r = 0;
    c = 0;

    while ( r < R ) {
        c = 0;
        while ( c < C ) {
            (*A)[r][c] =  randn(0,1) / sqrt( R );
            ++c;
        }
        ++r;
    }
    
    return 0;
}

numeric_t* get_random_vector(int L, int R) {
    
    int l = 0;
    numeric_t *p;
    p = e_calloc(L, sizeof(numeric_t));
    
    while ( l < L ) {
        p[l] = randn(0,1) / sqrt( R / 5 );
        ++l;
    }
    
    return p;
    
}

numeric_t** get_random_matrix(int R, int C)
{
    int r = 0, c = 0;
    numeric_t ** p;
    p = e_calloc(R, sizeof(numeric_t*));
    
    while ( r < R ) {
        p[r] = e_calloc(C, sizeof(numeric_t));
        ++r;
    }
    
    r = 0;
    c = 0;
    
    while ( r < R ) {
        c = 0;
        while ( c < C ) {
            p[r][c] =  _randf / sqrt( R / 2.0 );
            ++c;
        }
        ++r;
    }
    
    return p;
}

numeric_t**  get_zero_matrix(int R, int C)
{
    int r = 0;
    numeric_t ** p;
    p = e_calloc(R, sizeof(numeric_t*));
    
    while ( r < R )
    {
        p[r] = e_calloc(C, sizeof(numeric_t));
        ++r;
    }
    r = 0;
    while ( r < R )
    {
        memset(p[r], 0, C*sizeof(numeric_t));
        ++r;
    }
    return p;
}

int   init_zero_matrix(numeric_t*** A, int R, int C)
{
    int r = 0;
    *A = e_calloc(R, sizeof(numeric_t*));
    while ( r < R )
    {
        (*A)[r] = e_calloc(C, sizeof(numeric_t));
        ++r;
    }
    
    r = 0;
    while ( r < R )
    {
        memset((*A)[r], 0, C*sizeof(numeric_t));
        ++r;
    }
    return 0;
}

int   free_matrix(numeric_t** A, int R)
{
    int r = 0;
    while ( r < R ) {
        free(A[r]);
        ++r;
    }
    free(A);
    return 0;
}

int   init_zero_vector(numeric_t** V, int L)
{
    *V = e_calloc(L, sizeof(numeric_t));
    memset(*V, 0, L*sizeof(numeric_t));
    return 0;
}

numeric_t*   get_zero_vector(int L)
{
    numeric_t *p = e_calloc(L, sizeof(numeric_t));
    memset(p, 0, L*sizeof(numeric_t));
    return p;
}

int   free_vector(numeric_t** V)
{
    free(*V);
    *V = NULL;
    return 0;
}

void  copy_vector(numeric_t* A, numeric_t* B, int L)
{
    int l = 0;
    
    while ( l < L ) {
        A[l] = B[l];
        ++l;
    }
}

void  matrix_add(numeric_t** A, numeric_t** B, int R, int C)
{
    int r = 0, c = 0;
    
    while ( r < R ) {
        c = 0;
        while ( c < C ) {
            A[r][c] += B[r][c];
            ++c;
        }
        ++r;
    }
}

void  vector_set_to_zero(numeric_t* V, int L )
{
    memset(V, 0, L*sizeof(numeric_t));
}

void  matrix_set_to_zero(numeric_t** A, int R, int C)
{
    int r = 0;
    
    while ( r < R )
    {
        memset(A[r], 0, C*sizeof(numeric_t));
        ++r;
    }
}

void  matrix_substract(numeric_t** A, numeric_t** B, int R, int C)
{
    int r = 0, c = 0;
    
    while ( r < R ) {
        c = 0;
        while ( c < C ) {
            A[r][c] -= B[r][c];
            ++c;
        }
        ++r;
    }
}

void  matrix_scalar_multiply(numeric_t** A, numeric_t b, int R, int C)
{
    int r = 0, c = 0;
    
    while ( r < R ) {
        c = 0;
        while ( c < C ) {
            A[r][c] *= b;
            ++c;
        }
        ++r;
    }
}
void  matrix_clip(numeric_t** A, numeric_t limit, int R, int C)
{
    int r = 0, c = 0;
    
    while ( r < R ) {
        c = 0;
        while ( c < C ) {
            if ( A[r][c] > limit )
                A[r][c] = limit;
            else if ( A[r][c] < -limit )
                A[r][C] = -limit;
            ++c;
        }
        ++r;
    }
}

numeric_t one_norm(numeric_t* V, int L)
{
    int l = 0;
    numeric_t norm = 0.0;
    while ( l < L ) {
        norm += fabs(V[l]);
        ++l;
    }
    
    return norm;
}

int   vectors_fit(numeric_t* V, numeric_t limit, int L)
{
    int l = 0;
    int msg = 0;
    numeric_t norm = 0.0;
    while ( l < L )
    {
        if ( V[l] > limit || V[l] < -limit )
        {
            msg = 1;
            norm = one_norm(V, L);
            break;
        }
        ++l;
    }
    
    if ( msg && norm != 0.0 )
        vectors_mutliply_scalar(V, limit / norm, L);
    
    return msg;
}

int   vectors_clip(numeric_t* V, numeric_t limit, int L)
{
    int l = 0;
    int msg = 0;
    while ( l < L ) {
        if ( V[l] > limit ) {
            msg = 1;
            V[l] = limit;
        } else if ( V[l] < -limit ) {
            msg = 1;
            V[l] = -limit;
        }
        ++l;
    }
    
    return msg;
}

void  matrix_store(numeric_t ** A, int R, int C, FILE * fp)
{
    int r = 0, c = 0;
    size_t i = 0;
    char *p;
    
    while ( r < R ) {
        c = 0;
        while ( c < C ) {
            i = 0; p = (char*)&A[r][c];
            while ( i < sizeof(numeric_t) ) {
                fputc(*(p), fp);
                ++i; ++p;
            }
            ++c;
        }
        ++r;
    }
    
}

void  vector_print_min_max(char *name, numeric_t *V, int L)
{
    int l = 0;
    numeric_t min = 100;
    numeric_t max = -100;
    while ( l < L ) {
        if ( V[l] > max )
            max = V[l];
        if ( V[l] < min )
            min = V[l];
        ++l;
    }
    printf("%s min: %.10lf, max: %.10lf\n", name, min, max);
}

void  matrix_read(numeric_t ** A, int R, int C, FILE * fp)
{
    int r = 0, c = 0;
    size_t i = 0;
    char *p;
    numeric_t value;
    
    while ( r < R ) {
        c = 0;
        while ( c < C ) {
            i = 0; p = (char*)&value;
            while ( i < sizeof(numeric_t) ) {
                *(p) = fgetc(fp);
                ++i; ++p;
            }
            A[r][c] = value;
            ++c;
        }
        ++r;
    }
    
}

void  vector_store(numeric_t* V, int L, FILE * fp)
{
    int l = 0;
    size_t i = 0;
    char *p;
    
    while ( l < L ) {
        i = 0; p = (char*)&V[l];
        while ( i < sizeof(numeric_t) ) {
            fputc(*(p), fp);
            ++i; ++p;
        }
        ++l;
    }
}

void  vector_read(numeric_t * V, int L, FILE * fp)
{
    int l = 0;
    size_t i = 0;
    char *p;
    numeric_t value;
    
    while ( l < L ) {
        i = 0; p = (char*)&value;
        while ( i < sizeof(numeric_t) ) {
            *(p) = fgetc(fp);
            ++i; ++p;
        }
        V[l] = value;
        ++l;
    }
    
}

void  vector_store_ascii(numeric_t* V, int L, FILE * fp)
{
    int l = 0;
    
    while ( l < L ) {
        fprintf(fp, "%.20lf\r\n", V[l]);
        ++l;
    }
}

void  vector_read_ascii(numeric_t * V, int L, FILE * fp)
{
    int l = 0;
    const char * FORMAT = (sizeof(numeric_t) == sizeof(float)) ? "%f" : "%lf";
    
    while ( l < L ) {
        if ( fscanf(fp, FORMAT, &V[l]) <= 0 ) {
            fprintf(stderr, "%s.%s Failed to read file\r\n",
                    __FILE__, __func__);
            exit(1);
        }
        ++l;
    }
    
}

/*
 *   This function is used to store a JSON file representation
 *   of a LSTM neural network that can be read by an HTML application.
 */
void  vector_store_as_matrix_json(numeric_t* V, int R, int C, FILE * fp)
{
    int r = 0, c = 0;
    
    if ( fp == NULL )
        return; // No file, nothing to do.
    
    fprintf(fp, "[");
    
    r = 0;
    
    while ( r < R ) {
        
        if ( r > 0 )
            fprintf(fp, ",");
        
        fprintf(fp,"[");
        
        c = 0;
        while ( c < C ) {
            
            if ( c > 0 )
                fprintf(fp, ",");
            
            fprintf(fp,"%.15f", V[r*C + c]);
            
            ++c;
        }
        
        fprintf(fp,"]");
        
        ++r;
    }
    
    fprintf(fp, "]");
}


/*
 *   This function is used to store a JSON file representation
 *   of a LSTM neural network that can be read by an HTML application.
 */
void  vector_store_json(numeric_t* V, int L, FILE * fp)
{
    int l = 0;
    
    if ( fp == NULL )
        return; // No file, nothing to do.
    
    fprintf(fp, "[");
    
    while ( l < L ) {
        
        if ( l > 0 )
            fprintf(fp, ",");
        
        fprintf(fp,"%.15f", V[l]);
        
        ++l;
    }
    
    fprintf(fp, "]");
}

/*
 * Gaussian generator: https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
 */
numeric_t randn (numeric_t mu, numeric_t sigma)
{
    numeric_t U1, U2, W, mult;
    static numeric_t X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (numeric_t) X2);
    }
    
    do {
        U1 = -1 + _randf * 2;
        U2 = -1 + _randf * 2;
        W = pow (U1, 2) + pow (U2, 2);
    } while ( W >= 1 || W == 0 );
    
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (numeric_t) X1);
}

numeric_t sample_normal()
{
    numeric_t u = _randf * 2 - 1;
    numeric_t v = _randf * 2 - 1;
    numeric_t r = u * u + v * v;
    if (r == 0 || r > 1)
        return sample_normal();
    numeric_t c = sqrt(-2 * log(r) / r);
    return u * c;
}

/* Memory related utilities */
static size_t alloc_mem_tot = 0;
void*   e_calloc(size_t count, size_t size)
{
    void *p = calloc(count, size);
    if ( p == NULL ) {
        /* Failed to allocate this memory will exit */
        fprintf(stderr, "%s error: Failed to allocate %zu bytes, having allocated %zu in total already.\n",
                __func__, count*size, alloc_mem_tot);
        exit(1);
    }
    alloc_mem_tot += count*size;
    return p;
}

size_t  e_alloc_total()
{
    return alloc_mem_tot;
}

