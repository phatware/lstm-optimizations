/*
* TANF research perject
* LSTM Network quick test
* Author:  Stan Miasnikov
* (c) 2022
*/

#pragma once 

#include <iostream>
#include <fstream>
#include <string>

class FileProcessor {
public:
    FileProcessor();
    FileProcessor(const FileProcessor& orig);
    virtual ~FileProcessor();
    
    /**
     * 
     */
    std::vector<double> read(std::string fileName);
    /**
     * Read a text file containing comma separated values.
     * 
     * each vector contain variables in one row
     * last array vector is the target vector 
     */
    std::vector<double> * readMultivariate(std::string fileName, int lines, int variables, int * inputCols, int targetValCol);
    /**
     * 
     */
    int write(std::string fileName);
    /**
     * 
     */
    int append(std::string line);
    /**
     * 
     */
    int writeUniVariate(std::string fileName, std::string outFileName, int valuesPerLine, int columnIndx);
    
private:
    std::ofstream out_file;

};


