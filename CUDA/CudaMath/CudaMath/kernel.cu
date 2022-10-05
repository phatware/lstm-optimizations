
#define _INC_MATH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

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


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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


static map<const char*, double> _FUNCS{
    {"baseline", 0},
    {"plus", 0},
    {"minus", 0},
    {"mult", 0},
    {"div", 0},
    {"sqrt", 0},
    {"log", 0},
    {"exp", 0},
    {"elu", 0},
    {"relu", 0},
    {"leaky_relu", 0},
    {"sigmoid", 0},
    {"soft", 0},
    {"tanh", 0},
    {"tanf", 0},
};

std::string getEnvVar(std::string const& key)
{
    char* buff = NULL;
    size_t sz = 0;
    _dupenv_s(&buff, &sz, key.c_str());
    return buff == NULL ? std::string("") : std::string(buff);
}

#define randf() ((double) rand()) / ((double) (RAND_MAX))
#define FUNC_TEST(name, expr)                   \
    total = 0;                                  \
    srand(42);                                  \
    tmr.reset();                                \
    for (int64_t i = 0; i < 100000000; i++) {   \
        r1 = randf();                           \
        r2 = randf();                           \
        total += expr;                          \
    }                                           \
    double name = tmr.elapsed();                \
    _FUNCS[#name] += (name - baseline + 0.05);  \
    printf(#name);                              \
    printf(":\n%.7f", name - baseline + 0.05);  \
    printf("   %.7f\n", total);


int main()
{
    double total, r1, r2;
    Timer tmr;


    FUNC_TEST(baseline, 1.0);


    for (int j = 0; j < 10; j++)
    {
        // time various floating point operations.
        //   subtracts off the baseline time to give
        //   a better approximation of the cost
        //   for just the specified operation
        FUNC_TEST(plus, r1 + r2)
        FUNC_TEST(minus, r1 - r2)
        FUNC_TEST(mult, r1 * r2)
        FUNC_TEST(div, r1 / r2)
        FUNC_TEST(sqrt, std::sqrt(r1))
        FUNC_TEST(log, std::log(r1))
        FUNC_TEST(exp, std::exp(r1))
        FUNC_TEST(soft, soft(r1))
        FUNC_TEST(leaky_relu, leaky_relu(r1, 0.001))
        FUNC_TEST(relu, relu(r1))
        FUNC_TEST(elu, elu(r1))
        FUNC_TEST(sigmoid, sigmoid(r1))
        FUNC_TEST(tanh, std::tanh(r1))
        FUNC_TEST(tanf, tanf(r1))
    }

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

    printf("\n\nFunction,Relative Time\n");
    fout << "Function,Relative Time\n";
    for (map<const char*, double>::iterator it = _FUNCS.begin(); it != _FUNCS.end(); it++)
    {
        printf("%s,%.2f\n", it->first, 10.0 * it->second);
        fout << it->first << "," << 10.0 * it->second << "\n";
    }
    fout.close();

}

#if 0
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cu0daMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

#endif // 0
