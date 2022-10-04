/*
* TANF research perject
* LSTM Network quick test
* Author:  Stan Miasnikov
* (c) 2022
*/

#pragma once

#include <vector>

class DataProcessor {
public:
    DataProcessor();
    DataProcessor(const DataProcessor& orig);
    virtual ~DataProcessor();
    
    /**
     * normalize the given vector
     * @param vec vector
     * @param vecType vector type input or output
     * @return normalized vector
     */
    std::vector<double> process(std::vector<double> vec, int vecType); 
    /**
     * post process the given vector
     * @param vec vector
     * @return processed vector
     */
    std::vector<double> postprocess(std::vector<double> vec);
    /**
     * post process a given double
     * @param val
     * @return 
     */
    double postProcess(double val);
    
    /**
     * Print the given vector
     * @param vec vector
     * @return 
     */
    int printVector(std::vector<double> vec);
    
    
private:
    double out_magnitude = 0;
};


