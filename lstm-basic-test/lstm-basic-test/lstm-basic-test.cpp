/*
* TANF research perject
* LSTM Network quick test
* Author:  Stan Miasnikov
* (c) 2022
*/

#include <vector>
#include <chrono>
#include <math.h>
#include <memory.h>
#include "LSTMNet.h"
#include "DataProcessor.h"
#include "FileProcessor.h"

using namespace std::chrono;
// timer cribbed from
// https://gist.github.com/gongzhitaao/7062087
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return duration_cast<second_>
            (clock_::now() - beg_).count();
    }

private:
    typedef high_resolution_clock clock_;
    typedef duration<double, std::ratio<1>> second_;
    time_point<clock_> beg_;
};

typedef struct {
    double t1;
    double t2;
    double acc;
    int    tp;
    int    tn;
    int    fp;
    int    fn;
} history_t;

typedef struct __tanf_profiler {
    int     cellCount;
    history_t tanf;
    history_t tanh;
} tanf_profiler_t;

#if 0

static int univarPredicts(const char * inFile, const  char * outFile, bool useTanF) 
{
    int memCells = 5; // number of memory cells
    int trainDataSize = 300; // train data size
    int inputVecSize = 60; // input vector size
    int timeSteps = 60; // unfolded time steps
    double learningRate = 0.01;
    int predictions = 1300; // prediction points
    int iterations = 10; // training iterations with training data

    // Adding the time series in to a vector and preprocessing
    DataProcessor dataproc;
    FileProcessor fileProc;
    std::vector<double> timeSeries;

    timeSeries = fileProc.read(inFile);
    if (timeSeries.size() < 1)
    {
        std::cerr << "Unable to read training data from \"" << inFile << "\"" << std::endl;
        return -1;
    }
    timeSeries = dataproc.process(timeSeries, 1);

    // Creating the input vector Array
    std::vector<double>* input;
    input = new std::vector<double>[trainDataSize];
    std::vector<double> inputVec;

    for (int i = 0; i < trainDataSize; i++) 
    {
        inputVec.clear();
        for (int j = 0; j < inputVecSize; j++) 
        {
            inputVec.push_back(timeSeries.at(i + j));
        }
        inputVec = dataproc.process(inputVec, 0);
        input[i] = inputVec;
    }

    // Creating the target vector using the time series 
    std::vector<double>::const_iterator first = timeSeries.begin() + inputVecSize;
    std::vector<double>::const_iterator last = timeSeries.begin() + inputVecSize + trainDataSize;
    std::vector<double> targetVector(first, last);

    // Training the LSTM net
    LSTMNet lstm(memCells, inputVecSize, useTanF);
    lstm.train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);

    // Open the file to write the time series predictions
    std::ofstream out_file;
    // TODO: fix file path here
    out_file.open(outFile, std::ofstream::out | std::ofstream::trunc);

    std::vector<double> inVec;
    if (NULL != input)
        delete [] input;
    input = new std::vector<double>[1];
    double result;
    double expected;

    for (int i = 0; i < inputVecSize; i++) 
    {
        out_file << dataproc.postProcess(timeSeries.at(i)) << "\n";
    }

    // std::cout << std::fixed;

    for (int i = 0; i < predictions; i++) 
    {

        inVec.clear();
        for (int j = 0; j < inputVecSize; j++) 
        {
            inVec.push_back(timeSeries.at(i + j));
        }

        inVec = dataproc.process(inVec, 0);
        input[0] = inVec;

        result = lstm.predict(input);
        std::cout << std::endl << "result: " << result << std::endl;

        expected = timeSeries.at(i + inputVecSize + 1);
        //MSE += std::pow(expected-result,2);

        result = dataproc.postProcess(result);
        out_file << result - 20000 << "\n";
        std::cout << "result processed: " << result << std::endl << std::endl;
    }
    if (NULL != input)
        delete[] input;

    // std::cout << std::scientific;
    return 0;
}

#endif // 0

history_t multivarPredicts(bool useTanF,
                            const char * dataTrain, 
                            const char * dataTest, 
                            int memCells,
                            const char * outPath = NULL, 
                            bool zeroMargin = true)
{
    // Multivariate time series data prediction 

    int inputVecSize = 5; // input vector size
    int trainDataSize = 5000; // train data size
    int timeSteps = 1; // data points used for one forward step
    double learningRate = 0.0001;
    int iterations = 10; // training iterations with training data
    int lines = 5000;

    DataProcessor dataproc;
    FileProcessor fileProc;
    history_t  history = { 0 };

    int colIndxs[] = { 0, 0, 1, 1, 1, 1, 1 };
    int targetValCol = 7;

    std::vector<double>* timeSeries;
    timeSeries = fileProc.readMultivariate(dataTrain, lines, inputVecSize, colIndxs, targetValCol);
    if (NULL == timeSeries || timeSeries->size() < 1)
    {
        std::cerr << "Unable to read training data from \"" << dataTrain << "\"" << std::endl;
        return history;
    }

    // Creating the input vector Array
    std::vector<double>* input;
    input = new std::vector<double>[trainDataSize];
    for (int i = 0; i < trainDataSize; i++) 
    {
        input[i] = dataproc.process(timeSeries[i], 0);
    }

    // Creating the target vector using the time series 
    std::vector<double>::const_iterator first = timeSeries[lines].begin();
    std::vector<double>::const_iterator last = timeSeries[lines].begin() + trainDataSize;
    std::vector<double> targetVector(first, last);
    for (std::vector<double>::iterator it = targetVector.begin(); it != targetVector.end(); ++it) 
    {
        if (*it == 0)
            *it = -1;
    }

    // Training the LSTM net
    LSTMNet lstm(memCells, inputVecSize, useTanF);

    // measure train time
    Timer tm;
    tm.reset();
    lstm.train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);
    history.t1 = tm.elapsed();

    // Predictions
    int predictions = 2000; // prediction points
    lines = 2000; // lines read from the files

    if (NULL != timeSeries)
        delete [] timeSeries;
    timeSeries = fileProc.readMultivariate(dataTest, lines, inputVecSize, colIndxs, targetValCol);
    if (NULL == timeSeries || timeSeries->size() < 1)
    {
        std::cerr << "Unable to read test data from \"" << dataTest << "\"" << std::endl;
        return history;
    }
    
    if (NULL != input)
        delete [] input;
    input = new std::vector<double>[1];
    double result;
    double min = 0, max = 0;
    std::vector<double> resultVec;

    tm.reset();
    for (int i = 0; i < predictions; i++)
    {
        input[0] = dataproc.process(timeSeries[i], 0);
        result = lstm.predict(input);
        resultVec.push_back(result);

        if (i == 0)
        {
            min = result;
            max = result;
        }
        else 
        {
            if (min > result)
                min = result;
            if (max < result)
                max = result;
        }
    }
    history.t2 = tm.elapsed();
    
    double line = zeroMargin ? 0 : (min + max) / 2;

    int occu = 0;
    int notoccu = 0;
    int corr = 0;
    int incorr = 0;
    int corrNwMgn = 0;
    int incorrNwMgn = 0;

    // Open the file to write the time series predictions
    std::ofstream out_file;
    std::ofstream out_file2;

    // TODO: fix path for out files
    if (outPath != NULL)
    {
        std::string file1 = outPath;
        file1 += "/multiResults.txt";
        std::string file2 = outPath;
        file2 += "/multiTargets.txt";
        out_file.open(file1, std::ofstream::out | std::ofstream::trunc);
        out_file2.open(file2, std::ofstream::out | std::ofstream::trunc);
    }

    for (int i = 0; i < predictions; i++) 
    {
        if (outPath != NULL)
        {
            out_file << timeSeries[lines].at(i) << "," << resultVec.at(i) << "\n";
            out_file2 << timeSeries[lines].at(i) << ",";
            if (timeSeries[lines].at(i) == 1)
                out_file2 << 1 << "\n";
            else
                out_file2 << -1 << "\n";
        }

        if ((resultVec.at(i) > line) && (timeSeries[lines].at(i) == 1)) 
        {
            corr++;
            history.tp++;
            occu++;
        }
        else if ((resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 0)) 
        {
            corr++;
            history.tn++;
            notoccu++;
        }
        else if ((resultVec.at(i) <= line) && (timeSeries[lines].at(i) == 1)) 
        {
            incorr++;
            history.fn++;
            occu++;
        }
        else if ((resultVec.at(i) > line) && (timeSeries[lines].at(i) == 0)) 
        {
            incorr++;
            history.fp++;
            notoccu++;
        }


        if (line > 0)
        {
            if ((resultVec.at(i) <= line) && (resultVec.at(i) > 0))
            {
                if (timeSeries[lines].at(i) == 0)
                    corrNwMgn++;
                else
                    incorrNwMgn++;
            }
        }
        else
        {
            if ((resultVec.at(i) > line) && (resultVec.at(i) < 0))
            {
                if (timeSeries[lines].at(i) == 1)
                    corrNwMgn++;
                else
                    incorrNwMgn++;
            }
        }
    }

    if (outPath != NULL)
    {
        out_file.close();
        out_file2.close();
    }
    if (NULL != timeSeries)
        delete[] timeSeries;
    if (NULL != input)
        delete [] input;

    std::cout << std::endl;

    std::cout << "----------------------" << std::endl;
    std::cout << "Data " << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "Occupied    : " << occu << std::endl;
    std::cout << "NotOccupied : " << notoccu << std::endl << std::endl;

    std::cout << "----------------------" << std::endl;
    std::cout << "margin: " << line << " min: " << min << " max: " << max << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "Correct predictions   : " << corr << std::endl;
    std::cout << "Incorrect predictions : " << incorr << std::endl << std::endl;

    std::cout << "----------------------" << std::endl;
    std::cout << "Within the margin and 0" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "Correct: " << corrNwMgn << std::endl;
    std::cout << "Incorrect: " << incorrNwMgn << std::endl << std::endl << std::endl;

    std::cout << "True Positive  : " << history.tp << std::endl;
    std::cout << "True Negative  : " << history.tn << std::endl;
    std::cout << "False Positive : " << history.fp << std::endl;
    std::cout << "False Negative : " << history.fn << std::endl;

    history.acc = (corr / (double)predictions) * 100.0;

    std::cout << std::endl << "Accuracy: " << history.acc << "%" << std::endl << std::endl;

    std::cout << "----------------------" << std::endl;
    std::cout << "Timeouts: " << line << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "Train   : " << history.t1 << std::endl;
    std::cout << "Predict : " << history.t2 << std::endl;
    std::cout << "**********************" << std::endl << std::endl;

    return history;
}

inline void print_stat(const history_t* hist)
{
    std::cout << hist->t1 << ", ";
    std::cout << hist->t2 << ", ";
    std::cout << hist->acc << ", ";
    std::cout << hist->tp << ", ";
    std::cout << hist->tn << ", ";
    std::cout << hist->fp << ", ";
    std::cout << hist->fn;
}

int main(int argc, char ** argv) 
{
    // TODO: use command line parameters for data files

    // const char* dataTrain = "C:\\WORK\\ML\\tanf\\lstm-basic-test\\data\\datatraining.txt";
    // const char* dataTest  = "C:\\WORK\\ML\\tanf\\lstm-basic-test\\data\\datatest.txt";

    const char* dataTrain = "/home/stan/work/ML/tanf/lstm-basic-test/data/datatraining.txt";
    const char* dataTest  = "/home/stan/work/ML/tanf/lstm-basic-test/data/datatest.txt";

    // const char* dataTrain = "/Users/stan/work/ML/tanf/lstm-basic-test/data/datatraining.txt";
    // const char* dataTest  = "/Users/stan/work/ML/tanf/lstm-basic-test/data/datatest.txt";

    // TODO: also use command line
    int cells_default[] = { 10, 25, 50, 75, 150, 100, 200, 300, 500, 0 };
    int* ptr = cells_default;

    history_t tsF, tsH;
    std::vector<tanf_profiler_t *> history;

    while (0 != *ptr)
    {
        tanf_profiler_t* h = new tanf_profiler_t;
        h->cellCount = *ptr;
        tsF = multivarPredicts(true, dataTrain, dataTest, *ptr);
        if (tsF.t1 == 0)
            return -1;
        tsH = multivarPredicts(false, dataTrain, dataTest, *ptr);
        if (tsH.t1 == 0)
            return -1;
        memcpy(&h->tanf, &tsF, sizeof(history_t));
        memcpy(&h->tanh, &tsH, sizeof(history_t));
        history.push_back(h);
        ptr++;
    }

    std::cout << "Cells, Func, TrainTime, PredictTime, Accuracy, TP, TN, FP, FN" << std::endl;
    for (std::vector<tanf_profiler_t*>::iterator it = history.begin(); it != history.end(); ++it)
    {
        std::cout << (*it)->cellCount << ", tanf, ";
        print_stat(&(*it)->tanf);
        std::cout << std::endl;
        std::cout << (*it)->cellCount << ", tanh, ";
        print_stat(&(*it)->tanh);
        std::cout << std::endl; 
    }

    std::cout << std::endl;
    return 0;
}
