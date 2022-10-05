/*
* TANF research perject
* LSTM Network quick test
* Author:  Stan Miasnikov
* (c) 2022
*/

#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <ctime>

//#include <functional>


class LSTMNet {
public:
    LSTMNet(int memCells, int inputVecSize, bool newActivation);
    LSTMNet(const LSTMNet& orig);
    virtual ~LSTMNet();
    
    /**
    */
    int train(std::vector<double> * input, std::vector<double> output, int trainDataSize, int timeSteps, double learningRate, int iterations);
    /**
    * predict a future point using a input vector of size n
    * 
    * input: {t-n,...,t-2,t-1,t}
    * result: {t+1} 
    * 
    */
    double predict(std::vector<double> * input);

    
private:
    /**
     * Forward Propagation of the network
     *  
     */
    int forward(std::vector<double> * input, int timeSteps);
    /**
     * Backward Propagation of the network
     */
    int backward(std::vector<double> output, int timeSteps);
    /**
     * Initialize the weights and bias values for the gates
     * Random initialization
     */
    int initWeights();
    /**
     * Clear Vectors
     */
    int clearVectors();
    /**
     * print the given vector
     */
    int printVector(std::vector<double> vec);
    
private:
    int memCells = 0;
    int inputVectDim = 0;
    int timeSteps = 0;
    bool useTanF = false;
    
    // weight vector arrays
    std::vector<double> * aWeightVecArr = NULL;
    std::vector<double> * iWeightVecArr = NULL;
    std::vector<double> * fWeightVecArr = NULL;
    std::vector<double> * oWeightVecArr = NULL;
    
    // bias value arrays
    double * aBiasArr = NULL;
    double * iBiasArr = NULL;
    double * fBiasArr = NULL;
    double * oBiasArr = NULL;
    
    std::vector<double> * memCellOutArr = NULL;
    std::vector<double> * memCellStateArr = NULL;
    
    // gate output value arrays
    std::vector<double> * aGateVecArr = NULL;
    std::vector<double> * iGateVecArr = NULL;
    std::vector<double> * fGateVecArr = NULL;
    std::vector<double> * oGateVecArr = NULL;
    
    std::vector<double> * aGateDeltaVecArr = NULL;
    std::vector<double> * iGateDeltaVecArr = NULL;
    std::vector<double> * fGateDeltaVecArr = NULL;
    std::vector<double> * oGateDeltaVecArr = NULL;
    
    std::vector<double> * xDeltaVecArr = NULL;
    
    std::vector<double> DeltaOutVec;
    
    std::vector<double> * aDeltaWeightVecArr = NULL;
    std::vector<double> * iDeltaWeightVecArr = NULL;
    std::vector<double> * fDeltaWeightVecArr = NULL;
    std::vector<double> * oDeltaWeightVecArr = NULL;
    
    /**
     * data structures used for the prediction
     */    
    void deInit();
};
