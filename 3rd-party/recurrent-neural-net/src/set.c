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
#include "set.h"

void initialize_set(set_t * set)
{
    for ( int i = 0; i < SET_MAX_CHARS; i++ )
    {
        set->values[i] = '\0';
        set->free[i] = 1;
    }
}

int set_insert_symbol(set_t * set, char c)
{
    for (int i = 0; i <  SET_MAX_CHARS; i++)
    {
        if ( (char) set->values[i] == c && set->free[i] == 0 )
            return i;
        if ( set->free[i] )
        {
            set->values[i] = c;
            set->free[i] = 0;
            return 0;
        }
    }
    return -1;
}

char set_indx_to_char(set_t* set, int indx)
{
    if (indx >= SET_MAX_CHARS)
    {
        return '\0';
    }
    return (char) set->values[indx];
}

int set_char_to_indx(set_t* set, char c)
{
    for (int i = 0; i <  SET_MAX_CHARS; i++)
    {
        if ( set->values[i] == (int) c && set->free[i] == 0 )
            return i;
    }
    return -1;
}

int set_probability_choice(set_t* set, double* probs)
{
    double sum = 0, random_value;
    random_value = ((double) rand())/RAND_MAX;
    
    for (int i = 0; i < SET_MAX_CHARS; i++)
    {
        sum += probs[i];
        if ( sum - random_value > 0 )
            return set->values[i];
    }
    return 0;
}

int set_get_features(set_t* set)
{
    int i = 0;
    while ( set->free[i] == 0 )
        i++;
    if ( i < SET_MAX_CHARS )
        return i;
    return 0;
}

void set_print(set_t* set, double* probs)
{
    for (int i = 0; set->values[i] != 0 && i < SET_MAX_CHARS; i++)
    {
        if ( set->values[i] == '\n')
            printf("[ newline:  %lf ]\n", probs[i]);
        else
            printf("[ %c:     %lf ]\n", set->values[i], probs[i]);
    }
}

int set_greedy_argmax(set_t* set, double* probs)
{
    int max_i = 0;
    double max_double = 0.0;
    for (int i = 0; set->values[i] != 0 && i < SET_MAX_CHARS; i++)
    {
        if ( probs[i] > max_double )
        {
            max_i = i;
            max_double = probs[i];
        }
    }
    return set->values[max_i];
}

void set_store_as_json(set_t *set, FILE*fp)
{
    if ( fp == NULL )
        return;
    fprintf(fp, "{");
    for (int i = 0; set->values[i] != 0 && i < SET_MAX_CHARS; i++)
    {
        if ( i > 0 )
            fprintf(fp, ",");
        fprintf(fp, "\"%d\": \"%d\"", i, set->values[i]);
    }
    fprintf(fp, "}");
}

void set_store(set_t *set, FILE*fp)
{
    char * d;
    for (int i = 0; i < SET_MAX_CHARS; i++)
    {
        d = (char*) &set->values[i];
        for (size_t n = 0; n < sizeof(int); i++)
        {
            fputc(*d, fp);
            ++n;
        }
    }
}

int set_read(set_t *set, FILE*fp)
{
    int i = 0, c;
    unsigned int n;
    char *d,*e;
    
    while ( i < SET_MAX_CHARS )
    {
        d = (char*) &set->values[i];
        c = fgetc(fp);
        
        if ( c != EOF )
        {
            n = 0;
            e = (char*) &c;
            while ( n < sizeof(int) ) {
                d[n] = e[n];
                ++n;
            }
        }
        else
        {
            // The set was not read, it failed
            fprintf(stderr, "%s fail\n", __func__);
            return -1;
        }
        ++i;
    }
    return 0;
}

