//
//  set.c
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/10/22.
//

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

int set_probability_choice(set_t* set, numeric_t* probs)
{
    numeric_t sum = 0;
    numeric_t random_value = _randf;
    
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

void set_print(set_t* set, numeric_t* probs)
{
    for (int i = 0; set->values[i] != 0 && i < SET_MAX_CHARS; i++)
    {
        if ( set->values[i] == '\n')
            printf("[ newline:  %f ]\n", probs[i]);
        else
            printf("[ %c:     %f ]\n", set->values[i], probs[i]);
    }
}

int set_greedy_argmax(set_t* set, numeric_t* probs)
{
    int max_i = 0;
    numeric_t max_double = 0.0;
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

