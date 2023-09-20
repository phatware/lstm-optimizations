/*
* TANF research perject
* LSTM Network quick test
* Author:  Stan Miasnikov
* (c) 2022
*/

#include <vector>
#include <exception>
#include <sstream>

#include "FileProcessor.h"

FileProcessor::FileProcessor() 
{ 
}

FileProcessor::FileProcessor(const FileProcessor& orig) 
{ 
}

FileProcessor::~FileProcessor() 
{ 
}

std::vector<double> FileProcessor::read(std::string fileName) 
{
    std::vector<double> values;
    std::string line;
    std::ifstream file (fileName);
    int lineNo = 0;
    if (file.is_open())
    {
        while ( getline (file, line) )
        {
            lineNo++;
            try
            {
                values.push_back(std::stod(line));
            } 
            catch (std::exception& e)
            {
                // Note: do not show error for first line as it could be header
                if (1 < lineNo)
                    std::cerr << "Error in line  " << lineNo <<": " << e.what() << std::endl;
            }    
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file '" << fileName << "'";
    }
    return values;
}

std::vector<double> * FileProcessor::readMultivariate(std::string fileName, int lines, int variables, int * inputCols, int targetValCol) 
{
    std::string line;
    std::ifstream file (fileName);
    std::string token;
    int tokenNo = 0;
    int lineNo = 0;
    
    std::vector<double> target;
    std::vector<double> * data;
    data = new std::vector<double>[lines+1];

    int dataOffset = 0;
    
    if (file.is_open()) 
    {
        while ( getline (file,line) ) 
        {
            std::vector<double> input;
            lineNo++;
            try
            {
                std::stringstream ss(line);
                tokenNo = 0;
                while(std::getline(ss, token, ','))
                {
                    if (tokenNo == targetValCol)
                    {
                        target.push_back(std::stod(token));
                    }
                    else if (inputCols[tokenNo] == 1) 
                    {
                        input.push_back(std::stod(token));
                    }
                    tokenNo++;
                }
                data[dataOffset] = input;
            } 
            catch (std::exception& e)
            {
                if (1 == lineNo && target.size() == 0)
                {
                    // ignore this error -- header
                    continue;
                }
                std::vector<double> input (variables-1,0.0);
                data[dataOffset] = input;
                target.push_back(0.0);
                std::cerr << "Error in line " << lineNo << ": " << e.what() << std::endl;
            }    
            dataOffset++;
            if (dataOffset == lines)
                break;
        }
        file.close();
        data[lines] = target;
    }
    else
    {
        std::cerr << "Unable to open file '" << fileName << "'" << std::endl;
    }
    return data;
    
}

int FileProcessor::write(std::string fileName) 
{
    out_file.open(fileName,std::ofstream::out | std::ofstream::app);
    return 0;
}

int FileProcessor::append(std::string line) 
{
    out_file << "";
    return 0;
}

int FileProcessor::writeUniVariate(std::string fileName, std::string outFileName, int valuesPerLine, int columnIndx) 
{
    std::string line;
    std::ifstream file (fileName);
    std::string token;
    int tokenNo = 0;
    int lineNo = 0;
    
    std::ofstream out_file;
    out_file.open(outFileName,std::ofstream::out | std::ofstream::trunc);
    
    if (file.is_open())
    {
        if (valuesPerLine > 1) 
        {
            while ( getline (file,line) ) 
            {
                lineNo++;
                try{
                    std::stringstream ss(line);
                    tokenNo = 0;
                    while(std::getline(ss, token, ',')) 
                    {
                        if (tokenNo == columnIndx) 
                        {
                            out_file<<token<<"\n";
                        }
                        tokenNo++;
                    }
                } 
                catch (std::exception& e) 
                {
                    std::cerr << std::endl << "Error in line " << lineNo << ": " << e.what() << std::endl;
                }    
            }
        } 
        file.close();
    }
    else
    {
        std::cout << "Unable to open file '" << fileName << "'";
    }
    return 0;
}
