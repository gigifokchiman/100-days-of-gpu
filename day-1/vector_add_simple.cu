#include <stdio.h>
#include <cuda_runtime.h>  // CUDA-SPECIFIC: Header file for CUDA runtime API

// CUDA-SPECIFIC: __global__ declares a kernel function that runs on GPU
__global__ void addKernel(float *a, float *b, float *c, int n)
{
    // CUDA-SPECIFIC: threadIdx.x is a built-in variable giving thread's index
    int i = threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    // Standard C: Array declaration and initialization
    const int arraySize = 5;
    float a[arraySize] = {1, 2, 3, 4, 5};
    float b[arraySize] = {10, 20, 30, 40, 50};
    float c[arraySize] = {0};
    
    // Standard C: Pointer declaration (will point to GPU memory)
    float *dev_a, *dev_b, *dev_c;
    
    // CUDA-SPECIFIC: cudaMalloc allocates memory on GPU device
    cudaMalloc(&dev_a, arraySize * sizeof(float));
    cudaMalloc(&dev_b, arraySize * sizeof(float));
    cudaMalloc(&dev_c, arraySize * sizeof(float));
    
    // ========== CUDA-SPECIFIC: COPY DATA TO GPU ==========
    // cudaMemcpy transfers data between host (CPU) and device (GPU)
    // cudaMemcpyHostToDevice means copying from CPU to GPU
    cudaMemcpy(dev_a, a, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    
    // ========== CUDA-SPECIFIC: LAUNCH GPU KERNEL ==========
    // <<<1, arraySize>>> is kernel launch syntax
    // 1 = number of blocks, arraySize = threads per block
    addKernel<<<1, arraySize>>>(dev_a, dev_b, dev_c, arraySize);
    
    // ========== CUDA-SPECIFIC: COPY RESULTS BACK FROM GPU ==========
    // cudaMemcpyDeviceToHost means copying from GPU to CPU
    cudaMemcpy(c, dev_c, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Standard C: Print results
    printf("Results: ");
    for (int i = 0; i < arraySize; i++)
        printf("%.0f ", c[i]);
    printf("\n");
    
    // CUDA-SPECIFIC: cudaFree releases GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}