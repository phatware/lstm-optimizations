
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_utils.cuh"

#include <stdio.h>

extern "C" int rnn_main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus;
    int count = 0;

    cudaStatus = cudaGetDeviceCount(&count);
    if (cudaStatus != cudaSuccess || count < 1)
    {
        fprintf(stderr, "cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?\r\n");
        return -1;
    }
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\r\n");
        return -1;
    }

	int res = rnn_main(argc, argv);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return -1;
    }
    return res;
}
