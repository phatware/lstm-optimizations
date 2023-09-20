//
//  lstm_threaded.cpp
//  recurrent-neural-net
//
//  Created by Stan Miasnikov on 10/25/22.
//

#include "lstm_threaded.hpp"
#include <mutex>
#include <thread>
#include <vector>

class AdamOptimizerThread
{
public:
    AdamOptimizerThread(lstm_model_t* model, lstm_model_t* gradients, lstm_model_t* M, lstm_model_t* R)
    {
        _model = model;
        _gradients = gradients;
        _m_model = M;
        _r_model = R;
        
        beta1 = model->params->beta1;
        beta2 = model->params->beta2;
        
    }
    
    ~AdamOptimizerThread()
    {
        for ( int i = 0; i < LSTM_PARAMTERS; i++)
        {
            if (_threads[i].joinable())
                _threads[i].join();
        }
    }
    
    void optimize(unsigned int t)
    {
        beta1t = 1.0 / ( 1.0 - pow(beta1, t+1));
        beta2t = 1.0 / ( 1.0 - pow(beta2, t+1));

        for ( int i = 0; i < LSTM_PARAMTERS; i++)
        {
            int size1 = (i == LSTM_WB_Y) ? _model->Y * _model->N : _model->N * _model->S;
            int size2 = (i == LSTM_WB_Y) ? _model->Y : _model->N;
            _threads[i] = std::thread([this, i, size1, size2] {
                
                // will overwrite if file exists
                vectors_copy_multiply_scalar(_gradients->Wm[i], _gradients->W[i], 1.0 - beta1, size1);
                vectors_copy_multiply_scalar(_gradients->bm[i], _gradients->b[i], 1.0 - beta1, size2);
                vectors_multiply_scalar(_m_model->W[i], beta1, size1);
                vectors_multiply_scalar(_m_model->b[i], beta1, size2);
                vectors_add(_m_model->W[i], _gradients->W[i], size1);
                vectors_add(_m_model->b[i], _gradients->b[i], size2);
                // M Done!
                
                // Computing R
                vectors_multiply(_gradients->W[i], _gradients->W[i], size1);
                vectors_multiply(_gradients->b[i], _gradients->b[i], size2 );
                vectors_copy_multiply_scalar(_gradients->Wm[i], _gradients->W[i], 1.0 - beta2, size1);
                vectors_copy_multiply_scalar(_gradients->bm[i], _gradients->b[i], 1.0 - beta2, size2);
                vectors_multiply_scalar(_r_model->W[i], beta2, size1);
                vectors_multiply_scalar(_r_model->b[i], beta2, size2);
                vectors_add(_r_model->W[i], _gradients->W[i], size1);
                vectors_add(_r_model->b[i], _gradients->b[i], size2);
                // R done!
                
                vectors_copy_multiply_scalar(_m_model->Wm[i], _m_model->W[i], beta1t, size1);
                vectors_copy_multiply_scalar(_m_model->bm[i], _m_model->b[i], beta1t, size2);
                // M hat done!
                
                vectors_copy_multiply_scalar(_r_model->Wm[i], _r_model->W[i], beta2t, size1);
                vectors_copy_multiply_scalar(_r_model->bm[i], _r_model->b[i], beta2t, size2);
                // R hat done!
                
                vector_sqrt(_r_model->Wm[i], size1);
                vector_sqrt(_r_model->bm[i], size2);
                vectors_add_scalar(_r_model->Wm[i], EPSILON, size1);
                vectors_add_scalar(_r_model->bm[i], EPSILON, size2);
                vectors_copy_multiply_scalar(_gradients->Wm[i], _m_model->Wm[i], _model->params->learning_rate, size1);
                vectors_copy_multiply_scalar(_gradients->bm[i], _m_model->bm[i], _model->params->learning_rate, size2);
                vectors_div(_gradients->Wm[i], _r_model->Wm[i], size1);
                vectors_div(_gradients->bm[i], _r_model->bm[i], size2);
                vectors_subtract(_model->W[i], _gradients->Wm[i], size1);
                vectors_subtract(_model->b[i], _gradients->bm[i], size2);

            });
        }
    }
    
    
private:
    lstm_model_t * _model;
    lstm_model_t * _gradients;
    lstm_model_t * _m_model;
    lstm_model_t * _r_model;
    
    numeric_t beta1;
    numeric_t beta2;
    
    numeric_t beta1t;
    numeric_t beta2t;
    
    std::thread _threads[LSTM_PARAMTERS];

};

static void adam_optimize_layer(lstm_model_t* model, lstm_model_t* gradients, lstm_model_t* M, lstm_model_t* R, unsigned int t)
{
    AdamOptimizerThread adam(model, gradients, M, R);
    adam.optimize(t);
}

extern "C" void adam_optimize(int layers,
                              lstm_model_t** model_layers,
                              lstm_model_t** gradient_layers,
                              lstm_model_t** M_layers,
                              lstm_model_t** R_layers,
                              unsigned int n)
{
    if (layers < 2)
    {
        adam_optimize_layer(model_layers[0],
                            gradient_layers[0],
                            M_layers[0],
                            R_layers[0],
                            n);
    }
    else
    {
        std::thread threads[layers];

        for ( int p = 0; p < layers; p++ )
        {
            threads[p] = std::thread([model_layers, gradient_layers, M_layers, R_layers, n, p] {
                adam_optimize_layer(model_layers[p],
                                    gradient_layers[p],
                                    M_layers[p],
                                    R_layers[p],
                                    n);
            });
        }
        for ( int i = 0; i < layers; i++)
        {
            if (threads[i].joinable())
                threads[i].join();
        }
    }
}

