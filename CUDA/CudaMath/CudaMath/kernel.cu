
#define __CUDACC__ 1

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <map>
#include <string>
#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>

// cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

using namespace std;
using namespace std::chrono;

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
    typedef duration<double, ratio<1>> second_;
    time_point<clock_> beg_;
};


__global__ void tanfKernel(double* c, const double* a, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] / sqrt(1.0 + a[i] * a[i]);
}

__global__ void tanf2Kernel(double* c, const double* a, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
    {
        c[i] = __dmul_rn(a[i], a[i]);
        c[i] = __dadd_rn(1.0, c[i]);
        c[i] = __dsqrt_rn(c[i]);
        c[i] = __ddiv_rn(a[i], c[i]);
    }
}

__global__ void tanhKernel(double* c, const double* a, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = tanh(a[i]);
}

inline double tanf(double x)
{
    return x * sqrt(1.0 + x * x);
}

inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

inline double elu(double x)
{
    if (x > 0)
        return x;
    return exp(x) - 1.0;
}

inline double relu(double x)
{
    if (x > 0)
        return x;
    return 0.0;
}

inline double leaky_relu(double x, double a)
{
    if (x > 0)
        return x;
    return x * a;
}

inline double soft(double x)
{
    return x / (1.0 + abs(x));
}

std::string getEnvVar(std::string const& key)
{
    char* buff = NULL;
    size_t sz = 0;
    _dupenv_s(&buff, &sz, key.c_str());
    return buff == NULL ? std::string("") : std::string(buff);
}


#define VECTOR_SIZE     1000000
#define BLOCK_SIZE      1024
#define TEST_COUNT      1000

int main()
{
    Timer tmr;
    cudaError_t cudaStatus;
    double* vector = NULL;
    double* result = NULL;

    srand(42);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&vector, VECTOR_SIZE * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&result, VECTOR_SIZE * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cout << "Initializing vector, please wait..." << endl;

    int count = 0;
    for (double d = -20.0; d < 20.0; d += 0.0001)
    {
        cudaMemcpy(&vector[count], &d, sizeof(double), cudaMemcpyHostToDevice);
        count++;
    }

    cout << "Vector size " << count << "  Testing... " << endl;

    double elapsed1 = 0;
    double elapsed2 = 0;
    double* data1 = NULL;
    double* data2 = NULL;

    dim3 block(BLOCK_SIZE);
    dim3 grid((count / block.x) + 1);

    for (int i = 0; i < TEST_COUNT; i++)
    {
        // TANF
        tmr.reset();

        tanfKernel << <grid, BLOCK_SIZE >> > (result, vector, count);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "addKernel launch failed: %s\r\n", cudaGetErrorString(cudaStatus));
            return -1;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\r\n", cudaStatus);
            return -1;
        }
        elapsed1 += tmr.elapsed();

        if (NULL == data1)
        {
            data1 = new double[count];
            cudaMemcpy(data1, result, count * sizeof(double), cudaMemcpyDeviceToHost);
        }
        // TANH
        tmr.reset();

        tanhKernel << <grid, BLOCK_SIZE >> > (result, vector, count);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "addKernel launch failed: %s\r\n", cudaGetErrorString(cudaStatus));
            return -1;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\r\n", cudaStatus);
            return -1;
        }
        elapsed2 += tmr.elapsed();
 
        if (NULL == data2)
        {
            data2 = new double[count];
            cudaMemcpy(data2, result, count * sizeof(double), cudaMemcpyDeviceToHost);
        }
    }

    cout << elapsed1 / TEST_COUNT << " vs " << elapsed2 / TEST_COUNT << endl;

    std::ofstream fout;
    std::stringstream sstr;

#if _WIN32 || _WIN64
    cout << endl << getEnvVar("PROCESSOR_ARCHITECTURE") << " " << getEnvVar("PROCESSOR_IDENTIFIER");
    cout << " " << "MS C++ " << _MSC_VER << endl;

    sstr << getEnvVar("PROCESSOR_ARCHITECTURE") << " " << getEnvVar("PROCESSOR_IDENTIFIER") << " MS C++ " << _MSC_VER << ".csv";
#else
    cout << endl << __VERSION__ << endl;
#if __APPLE__
    const NXArchInfo* info = NXGetLocalArchInfo();
    cout << info->name << "." << info->cputype << "." << info->cpusubtype << endl;
    cout << info->description << endl;
    sstr << info->name << "." << info->cputype << "." << info->cpusubtype << " " << __VERSION__ << ".csv";
#endif
#if __linux__
    utsname result;      // declare the variable to hold the result
    uname(&result);      // call the uname() function to fill the struct
    std::cout << result; // show the result using the helper function

    sstr << result.machine << " " << result.sysname << " " << result.release << " " __VERSION__ << ".csv";
#endif
#endif // WIN

    fout.open(sstr.str());
    fout << "func1,func2\n";
    for (int i = 0; i < count; i++)
    {
        fout << data1[i] << "," << data2[i] << "\n";
    }
    fout.close();

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return -1;
    }
    return 0;
}
