
#define __CUDACC__

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

__global__ void eqKernel(double* c, const double* a, const double* b)
{
    int i = threadIdx.x;
    c[i] = a[i];
}

__global__ void addKernel(double *c, const double*a, const double *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void subKernel(double* c, const double* a, const double* b)
{
    int i = threadIdx.x;
    c[i] = a[i] - b[i];
}

__global__ void milKernel(double* c, const double* a, const double* b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

__global__ void divKernel(double* c, const double* a, const double* b)
{
    int i = threadIdx.x;
    c[i] = a[i] / b[i];
}

__global__ void tanfKernel(double* c, const double* a, const double* b)
{
    int i = threadIdx.x;
    c[i] = a[i] / __dsqrt_rn(1.0 + a[i] * a[i]);
}

__global__ void tanf2Kernel(double* c, const double* a, const double* b)
{
    int i = threadIdx.x;
    c[i] = __dmul_rn(a[i], a[i]);
    c[i] = __dadd_rn(1.0, c[i]);
    c[i] = __dsqrt_rn(c[i]);
    c[i] = __ddiv_rn(a[i], c[i]);
}

__global__ void tanhKernel(double* c, const double* a, const double* b)
{
    int i = threadIdx.x;
    c[i] = tanhf(a[i]);
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
    {"tanf2", 0},
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
    tmr.reset();                                \
    for (int64_t i = 0; i < 1000000; i++) {   \
        a[0] = randf();                           \
        a[1] = randf();                           \
        a[2] = 0;       \
cudaStatus = cudaMemcpy(r, a, 3 * sizeof(double), cudaMemcpyHostToDevice); \
if (cudaStatus != cudaSuccess) \
{ \
    fprintf(stderr, "cudaMemcpy failed!"); \
    return -1; \
} \
        expr<<<1, 1>>>(&r[2], &r[0], &r[1]); \
    }                                           \
cudaStatus = cudaDeviceSynchronize(); \
if (cudaStatus != cudaSuccess) { \
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); \
     return -1; \
} \
cudaStatus = cudaMemcpy(a, r, 3 * sizeof(double), cudaMemcpyDeviceToHost); \
if (cudaStatus != cudaSuccess) { \
    fprintf(stderr, "cudaMemcpy failed!"); \
    return -1; \
} \
    double name = tmr.elapsed();                \
    _FUNCS[#name] += (name - baseline + 0.05);  \
    printf(#name);                              \
    printf(":\n%.7f", name - baseline + 0.05);  \
    printf("   r1=%.5f r2=%.5f  t=%.5f\n", a[0], a[1], a[2]); \
    fflush(stdout);


int main()
{
    double total, a[3] = { 0 };
    Timer tmr;

    cudaError_t cudaStatus;

    double* r = NULL;

    srand(42);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&r, 3 * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    FUNC_TEST(baseline, eqKernel);

    for (int j = 0; j < 10; j++)
    {
        // time various floating point operations.
        //   subtracts off the baseline time to give
        //   a better approximation of the cost
        //   for just the specified operation
        FUNC_TEST(plus, addKernel)
        FUNC_TEST(minus, subKernel)
        FUNC_TEST(mult, milKernel)
        FUNC_TEST(div, divKernel)
        
        // FUNC_TEST(sqrt, std::sqrt(r[0]))
        //FUNC_TEST(log, std::log(r[0]))
        //FUNC_TEST(exp, std::exp(r[0]))
        //FUNC_TEST(soft, soft(r[0]))
        //FUNC_TEST(leaky_relu, leaky_relu(r[0], 0.001))
        //FUNC_TEST(relu, relu(r[0]))
        //FUNC_TEST(elu, elu(r[0]))
        //FUNC_TEST(sigmoid, sigmoid(r[0]))

        FUNC_TEST(tanh, tanhKernel)
        FUNC_TEST(tanf, tanfKernel)
        FUNC_TEST(tanf2, tanf2Kernel)
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

    cudaFree(r);
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return -1;
    }
    return 0;
}
