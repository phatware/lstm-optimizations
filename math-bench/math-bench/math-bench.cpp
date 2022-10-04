// math-bench.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <math.h>
#include <chrono>
#include <stdio.h>
#include <map>
#include <string>
#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>

#if _WIN32 || _WIN64
#include <Windows.h>

std::string getEnvVar(std::string const& key)
{
    char* buff = NULL;
    size_t sz = 0;
    _dupenv_s(&buff, &sz, key.c_str());
    return buff == NULL ? std::string("") : std::string(buff);
}

#else

#include <unistd.h>

#if __linux__
#include <sys/utsname.h>
#elif __APPLE__
#include <mach-o/arch.h>
#endif
#endif //

using namespace std;
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
    typedef duration<double, ratio<1>> second_;
    time_point<clock_> beg_;
};

// Activation functions 

inline double tanf(double x)
{
    return x / sqrt(1.0 + x * x);
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

#if __linux__
// a small helper to display the content of an utsname struct:
std::ostream& operator<<(std::ostream& os, const utsname& u) {
    return os << "sysname : " << u.sysname << '\n'
        << "nodename: " << u.nodename << '\n'
        << "release : " << u.release << '\n'
        << "version : " << u.version << '\n'
        << "machine : " << u.machine << '\n';
}
#endif // __linux__


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

int main(int argc, char** argv)
{
    double total, r1, r2;
    Timer tmr;

#if _WIN32 || _WIN64

    Sleep(1000);
#else
    sleep(1);
#endif

    FUNC_TEST(baseline, 1.0);

#if _WIN32 || _WIN64
    Sleep(1000);
#else
    sleep(1);
#endif

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
        FUNC_TEST(sqrt, sqrt(r1))
        FUNC_TEST(log, log(r1))
        FUNC_TEST(exp, exp(r1))
        FUNC_TEST(soft, soft(r1))
        FUNC_TEST(leaky_relu, leaky_relu(r1, 0.001))
        FUNC_TEST(relu, relu(r1))
        FUNC_TEST(elu, elu(r1))
        FUNC_TEST(sigmoid, sigmoid(r1))
        FUNC_TEST(tanh, tanh(r1))
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
