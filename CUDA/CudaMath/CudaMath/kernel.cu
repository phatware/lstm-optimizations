
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
#define VECTOR_SIZE     1000000
#define BLOCK_SIZE      1024
#define TEST_COUNT      5000

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

__global__ void tanfKernel(float* c, const float* a, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] / sqrtf(1 + a[i] * a[i]);
}

__global__ void tanhKernel(float* c, const float* a, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = tanhf(a[i]);
}

#if 0
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

inline double softsign(double x)
{
    return x / (1.0 + abs(x));
}
#endif // 0

std::string getEnvVar(std::string const& key)
{
    char* buff = NULL;
    size_t sz = 0;
    _dupenv_s(&buff, &sz, key.c_str());
    return buff == NULL ? std::string("") : std::string(buff);
}

template<typename T>
cudaError_t test_performance(double &elapsed1, double& elapsed2,T** data1, T** data2,int &count)
{
    Timer tmr;
    cudaError_t cudaStatus;
    T* vector = NULL;
    T* result = NULL;
    elapsed1 = 0;
    elapsed2 = 0;

    cudaStatus = cudaMalloc((void**)&vector, VECTOR_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&result, VECTOR_SIZE * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        cudaFree(vector);
        return cudaStatus;
    }

    cout << "Initializing vector, please wait..." << endl;

    count = 0;
    for (T d = -10.0f; d < 10.0f; d += 0.00002f)
    {
        cudaMemcpy(&vector[count], &d, sizeof(T), cudaMemcpyHostToDevice);
        count++;
        if (count >= VECTOR_SIZE)
            break;
    }
    dim3 block(BLOCK_SIZE);
    dim3 grid((count / block.x) + 1);

    cout << "Vector size " << count << "  Testing... " << endl << flush;

    for (int i = 0; i < TEST_COUNT; i++)
    {
        // TANF
        tmr.reset();

        tanfKernel << <grid, BLOCK_SIZE >> > (result, vector, count);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            goto error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            goto error;
        }
        elapsed1 += tmr.elapsed();

        if (NULL != data1 && NULL == *data1)
        {
            *data1 = new T[count];
            cudaMemcpy(*data1, result, count * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemset(result, 0, VECTOR_SIZE * sizeof(T));
        }
        // TANH
        tmr.reset();

        tanhKernel << <grid, BLOCK_SIZE >> > (result, vector, count);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            goto error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            goto error;
        }
        elapsed2 += tmr.elapsed();

        if (NULL != data2 && NULL == *data2)
        {
            *data2 = new T[count];
            cudaStatus = cudaMemcpy(*data2, result, count * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemset(result, 0, VECTOR_SIZE * sizeof(T));
        }
    }
error:
    cudaFree(result);
    cudaFree(vector);
    return cudaStatus;
}

int main()
{
    cudaError_t cudaStatus;
    double* ddata1 = NULL;
    double* ddata2 = NULL;
    float * fdata1 = NULL;
    float * fdata2 = NULL;
    double  elapsed1 = 0;
    double  elapsed2 = 0;
    int count = 0;

    srand(42);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
        return -1;
    }

    cout << "Testing single precision (float)" << endl << endl << flush;
    cudaStatus = test_performance<float>(elapsed1, elapsed2, &fdata1, &fdata2, count);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "CUDA failed: " << cudaGetErrorString(cudaStatus) << endl;
        return -1;
    }
    cout << "float tanf: " << elapsed1 / TEST_COUNT << " vs tanh: " << elapsed2 / TEST_COUNT << endl;

    cout << endl << "Testing double precision (double)" << endl << endl << flush;
    cudaStatus = test_performance<double>(elapsed1, elapsed2, &ddata1, &ddata2, count);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "CUDA failed: " << cudaGetErrorString(cudaStatus) << endl;
        return -1;
    }
    cout << "double tanf: " << elapsed1 / TEST_COUNT << " vs tanh: " << elapsed2 / TEST_COUNT << endl;

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
    fout << "dtanf,dtanh,ftanf,ftanh\n";
    for (int i = 0; i < count; i++)
    {
        fout << ddata1[i] << "," << ddata2[i] << "," << fdata1[i] << "," << fdata2[i] << "\n";
    }
    fout.close();

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        cerr << "cudaDeviceReset failed!" << endl;
        return -1;
    }
    return 0;
}
