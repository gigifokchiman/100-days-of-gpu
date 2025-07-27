#include <stdio.h>              // Standard C header
#include <cuda_runtime.h>       // CUDA-SPECIFIC: CUDA runtime API header

// CUDA-SPECIFIC: __global__ declares a GPU kernel function
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    // CUDA-SPECIFIC: Built-in variables for thread indexing
    // blockDim.x = threads per block, blockIdx.x = block index, threadIdx.x = thread index within block
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    // ========== PRODUCTION FEATURE: Dynamic Array Size ==========
    // Can handle any size, not just small fixed arrays
    int numElements = 1000000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);
    
    // Standard C: Allocate host (CPU) memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // ========== PRODUCTION FEATURE: NULL Check for Host Memory ==========
    // Essential for catching out-of-memory conditions
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize with sequential values for consistent performance testing
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = (float)i;        // Sequential values for predictable performance
        h_B[i] = (float)(i * 2);
    }
    
    // Device (GPU) memory pointers
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    // CUDA-SPECIFIC: Error handling type
    cudaError_t err = cudaSuccess;
    
    // ========== CUDA EVENTS FOR PRECISE GPU TIMING ==========
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    // ========== PRODUCTION FEATURE: Error Checking for GPU Allocation ==========
    // CUDA-SPECIFIC: Allocate memory on GPU device
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        // CUDA-SPECIFIC: cudaGetErrorString converts error code to readable string
        // Provides detailed error info (e.g., "out of memory", "invalid value")
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    // ========== TIMING: Start GPU measurement ==========
    cudaEventRecord(start_event);
    
    // CUDA-SPECIFIC: Copy data from host (CPU) to device (GPU)
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // ========== PRODUCTION FEATURE: Scalable Thread Organization ==========
    // CUDA thread organization for handling arrays of any size
    int threadsPerBlock = 256;  // Common choice for good occupancy
    // Calculate grid size to ensure all elements are processed
    // The +threadsPerBlock-1 ensures we round up (ceiling division)
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // CUDA-SPECIFIC: <<<blocks, threads>>> kernel launch syntax
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // ========== PRODUCTION FEATURE: Kernel Launch Error Detection ==========
    // CUDA-SPECIFIC: Check for kernel launch errors
    // Kernels launch asynchronously, so errors might not appear immediately
    err = cudaGetLastError();
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // CUDA-SPECIFIC: Copy results from device (GPU) to host (CPU)
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // ========== TIMING: End GPU measurement ==========
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_event, stop_event);
    
    // ========== PRODUCTION FEATURE: Result Verification ==========
    // Validate that GPU computation matches expected results
    // Essential for catching numerical errors or kernel bugs
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)  // Floating point tolerance
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    printf("GPU computation time: %.3f ms (%.0f elements)\n", gpu_time, (float)numElements);
    printf("✅ Test PASSED - All %d elements computed correctly!\n", numElements);
    
    // CUDA-SPECIFIC: Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Clean up CUDA events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    // Standard C: Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done\n");
    return 0;
}