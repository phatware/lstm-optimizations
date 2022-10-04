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
    * 
    * @param input: training data set
    * @param output: target values
    * @param trainDataSize: training data size
    * @param timeSteps: unfolding time steps
    * @param learningRate
    * @param iterations: training iterations
    * @return 0
    */
    int train(std::vector<double> * input, std::vector<double> output, int trainDataSize, int timeSteps, double learningRate, int iterations);
    /**
    * predict a future point using a input vector of size n
    * 
    * input: {t-n,...,t-2,t-1,t}
    * result: {t+1} 
    * 
    * @param input: input vector for he prediction
    * @return result: predicted value
    */
    double predict(std::vector<double> * input);

    
private:
    /**
     * Forward Propagation of the network
     *  
     * @param input: input vector
     * @param timeSteps: unfolded time steps in the input vector
     * @return 0
     */
    int forward(std::vector<double> * input, int timeSteps);
    /**
     * Backward Propagation of the network
     * 
     * @param output: output from the forward propagation
     * @param timeSteps: unfolded time steps
     * @return 0
     */
    int backward(std::vector<double> output, int timeSteps);
    /**
     * Initialize the weights and bias values for the gates
     * Random initialization
     * 
     * @return 0
     */
    int initWeights();
    /**
     * Clear Vectors
     * 
     * @return 0
     */
    int clearVectors();
    /**
     * print the given vector
     * 
     * @param vec: vector
     * @return 0 
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
