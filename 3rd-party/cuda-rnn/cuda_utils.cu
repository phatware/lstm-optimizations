
 #define __CUDACC__ 1

#include <stdio.h>
#include "cuda_utils.cuh"

#define BLOCK_SIZE  1000

__global__ void addKernel(double* c, const double* a, const double* b, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[i];
}

__global__ void subKernel(double* c, const double* a, const double* b, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] - b[i];
}

__global__ void milKernel(double* c, const double* a, const double* b, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] * b[i];
}

__global__ void divKernel(double* c, const double* a, const double* b, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] / b[i];
}

__global__ void mulScalarKernel(double* c, const double* a, const double* b, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] * b[0];
}

__global__ void addScalarKernel(double* c, const double* a, const double* b, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[0];
}

__global__ void addMulScalarKernel(double* c, const double* a, const double* b, const double* s, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[i] * s[0];
}

__global__ void subMulScalarKernel(double* c, const double* a, const double* b, const double* s, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] - b[i] * s[0];
}

__global__ void sqrtKernel(double* c, const double* a, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = (a[i]);
}

__global__ void tanfKernel(double* y, const double* x, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        y[i] = x[i] / __dsqrt_rn(1.0 + x[i] * x[i]);
}

__global__ void tanhKernel(double* y, const double* x, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        y[i] = tanhf(x[i]);
}

__global__ void sigmoidKernel(double* y, const double* x, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        y[i] = 1.0 / (1.0 + exp(-x[i]));
}

__global__ void dtanfKernel(double* dx, const double* y, const double* dy, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
    {
        double t = (1.0 - y[i] * y[i]);
        dx[i] = dy[i] * sqrt(t * t * t);
    }
}

__global__ void dtanhKernel(double* dx, const double* y, const double* dy, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        dx[i] = (1.0 - y[i] * y[i]) * dy[i];
}

__global__ void dsigmoidKernel(double* dx, const double* y, const double* dy, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        dx[i] = (1.0 - y[i]) * y[i] * dy[i];
}

class CcudaVectorBlock
{
public:
    CcudaVectorBlock()
    {
    }

    ~CcudaVectorBlock()
    {
        cuda_free_memory();
    }

    cudaError_t cuda_inplace_vectors_math_op(
        double* a,
        const double* b,
        unsigned int size,
        double scalar,
        vector_math_op_t operation);

    cudaError_t cuda_forward_vectors_math_op(
        double* y,
        const double* x,
        unsigned int size,
        vector_forward_op_t operation);

    cudaError_t cuda_backward_vectors_math_op(
        double* dx,
        const double* y,
        const double* dy,
        unsigned int size,
        vector_backward_op_t operation);

private:
    double* dev_a = NULL;
    double* dev_b = NULL;
    double* dev_c = NULL;
    double* dev_s = NULL;
    unsigned int current_size = 0;

    cudaError_t cuda_init_memory(unsigned int size);
    void cuda_free_memory()
    {
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_s);

        dev_a = NULL;
        dev_b = NULL;
        dev_c = NULL;
        dev_s = NULL;
        current_size = 0;
    }
};

CcudaVectorBlock vectorMem;

cudaError_t CcudaVectorBlock::cuda_init_memory(unsigned int size)
{
    if (size <= current_size)
        return cudaSuccess;

    cuda_free_memory();
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaError_t cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\r\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\r\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\r\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_s, sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\r\n");
        goto Error;
    }

    current_size = size;
    return cudaStatus;

Error:
    cuda_free_memory();
    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t CcudaVectorBlock::cuda_inplace_vectors_math_op(
    double* a,
    const double* b,
    unsigned int size,
    double scalar,
    vector_math_op_t operation)
{
    cudaError_t cudaStatus;

    dim3 block(BLOCK_SIZE);
    dim3 grid((size / block.x) + 1);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cuda_init_memory(size);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!\r\n");
        return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!\r\n");
        return cudaStatus;
    }

    if (NULL != b)
    {
        cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy failed!\r\n");
            return cudaStatus;
        }
    }
    if (0 != scalar)
    {
        cudaStatus = cudaMemcpy(dev_s, &scalar, sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy failed!\r\n");
            return cudaStatus;
        }
    }
    switch (operation)
    {
        case vector_math_multiply :
            if (NULL == b)
            {
                fprintf(stderr, "Second vector is required!\r\n");
                return cudaStatus;
            }
            milKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, dev_b, size);
            break;

        case vector_math_divide:
            if (NULL == b)
            {
                fprintf(stderr, "Second vector is required!\r\n");
                return cudaStatus;
            }
            divKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, dev_b, size);
            break;

        case vector_math_add:
            if (NULL == b)
            {
                fprintf(stderr, "Second vector is required!\r\n");
                return cudaStatus;
            }
            addKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, dev_b, size);
            break;

        case vector_math_subtract :
            if (NULL == b)
            {
                fprintf(stderr, "Second vector is required!\r\n");
                return cudaStatus;
            }
            subKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, dev_b, size);
            break;

        case vector_math_scalar_multiply :
            if (NULL == scalar)
            {
                fprintf(stderr, "Non-zero scalar is required!\r\n");
                return cudaStatus;
            }
            mulScalarKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, dev_s, size);
            break;

        case vector_math_scalar_add :
            if (NULL == scalar)
            {
                fprintf(stderr, "Non-zero scalar is required!\r\n");
                return cudaStatus;
            }
            addScalarKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, dev_s, size);
            break;

        case vector_math_add_scalar_multiply:
            if (NULL == scalar)
            {
                fprintf(stderr, "Non-zero scalar is required!\r\n");
                return cudaStatus;
            }
            if (NULL == b)
            {
                fprintf(stderr, "Second vector is required!\r\n");
                return cudaStatus;
            }
            addMulScalarKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, dev_b, dev_s, size);
            break;

        case vector_math_subtract_scalar_multiply :
            if (NULL == scalar)
            {
                fprintf(stderr, "Non-zero scalar is required!\r\n");
                return cudaStatus;
            }
            if (NULL == dev_b)
            {
                fprintf(stderr, "Second vector is required!\r\n");
                return cudaStatus;
            }
            subMulScalarKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, dev_b, dev_s, size);
            break;

        case vector_math_sqrt :
            sqrtKernel << <grid, BLOCK_SIZE >> > (dev_c, dev_a, size);
            break;
    }

    // Launch a kernel on the GPU with one thread for each element.

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\r\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\r\n", cudaStatus);
        return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!\r\n");
        return cudaStatus;
    }
    return cudaStatus;
}

cudaError_t CcudaVectorBlock::cuda_forward_vectors_math_op(
    double* y, 
    const double* x, 
    unsigned int size, 
    vector_forward_op_t operation)
{
    cudaError_t cudaStatus;

    dim3 block(BLOCK_SIZE);
    dim3 grid((size / block.x) + 1);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cuda_init_memory(size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\r\n");
        return cudaStatus;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_b, x, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    switch (operation)
    {
        case vector_forward_tanh:
            tanhKernel << <grid, BLOCK_SIZE >> > (dev_a, dev_b, size);
            break;

        case vector_forward_tanf:
            tanfKernel << <grid, BLOCK_SIZE >> > (dev_a, dev_b, size);
            break;

        case vector_forward_sigmoid:
            sigmoidKernel << <grid, BLOCK_SIZE >> > (dev_a, dev_b, size);
            break;
    }

    // Launch a kernel on the GPU with one thread for each element.

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(y, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

cudaError_t CcudaVectorBlock::cuda_backward_vectors_math_op(
    double* dx, 
    const double* y, 
    const double* dy, 
    unsigned int size, 
    vector_backward_op_t operation)
{
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaError_t cudaStatus;

    dim3 block(BLOCK_SIZE);
    dim3 grid((size / block.x) + 1);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cuda_init_memory(size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\r\n");
        return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_c, dy, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(dev_b, y, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    switch (operation)
    {
    case vector_forward_tanh:
        dtanhKernel << <grid, BLOCK_SIZE >> > (dev_a, dev_b, dev_c, size);
        break;

    case vector_forward_tanf:
        dtanfKernel << <grid, BLOCK_SIZE >> > (dev_a, dev_b, dev_c, size);
        break;

    case vector_forward_sigmoid:
        dsigmoidKernel << <grid, BLOCK_SIZE >> > (dev_a, dev_b, dev_c, size);
        break;
    }

    // Launch a kernel on the GPU with one thread for each element.

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(dx, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

// External "C" Interface

// Helper function for using CUDA to add vectors in parallel.
extern "C" cudaError_t cuda_inplace_vectors_math_op(
    double* a,
    const double* b,
    unsigned int size,
    double scalar,
    vector_math_op_t operation)
{
    return vectorMem.cuda_inplace_vectors_math_op(a, b, size, scalar, operation);
}

extern "C"  cudaError_t cuda_forward_vectors_math_op(
    double* y,
    const double* x,
    unsigned int size,
    vector_forward_op_t operation)
{
    return vectorMem.cuda_forward_vectors_math_op(y, x, size, operation);
}

extern "C" cudaError_t cuda_backward_vectors_math_op(
    double* dx,
    const double* y,
    const double* dy,
    unsigned int size,
    vector_backward_op_t operation)
{
    return vectorMem.cuda_backward_vectors_math_op(dx, y, dy, size, operation);
}
