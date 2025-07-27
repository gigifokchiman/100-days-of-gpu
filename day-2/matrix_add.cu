#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA-SPECIFIC: Matrix addition kernel using 2D thread blocks
__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height)
{
    // Calculate global thread indices
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Check bounds
    if (col < width && row < height)
    {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

int main(void)
{
    // ========== PRODUCTION FEATURE: Dynamic Matrix Size ==========
    // Can handle large matrices, not just small fixed sizes
    int width = 1024;   // Matrix width
    int height = 1024;  // Matrix height
    int numElements = width * height;
    size_t size = numElements * sizeof(float);
    
    printf("[Matrix addition of %dx%d matrices]\n", width, height);
    
    // Allocate host (CPU) memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // ========== PRODUCTION FEATURE: NULL Check for Host Memory ==========
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize matrices with sequential values
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }
    
    // Device (GPU) memory pointers
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaError_t err = cudaSuccess;
    
    // ========== CUDA EVENTS FOR PRECISE GPU TIMING ==========
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    // ========== PRODUCTION FEATURE: Error Checking for GPU Allocation ==========
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy matrices from host to device
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // ========== PRODUCTION FEATURE: 2D Thread Organization for Matrices ==========
    // Using 2D blocks for better spatial locality and easier indexing
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    
    // ========== TIMING: Start GPU measurement (kernel only) ==========
    cudaEventRecord(start_event);
    
    // Launch the Matrix Addition CUDA Kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matrixAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // ========== TIMING: End GPU measurement ==========
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_event, stop_event);
    
    // Copy result matrix from device to host
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // ========== PRODUCTION FEATURE: Result Verification ==========
    // Verify that the result is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            fprintf(stderr, "Expected: %f, Got: %f\n", h_A[i] + h_B[i], h_C[i]);
            exit(EXIT_FAILURE);
        }
    }
    
    printf("GPU computation time: %.3f ms (%d elements)\n", gpu_time, numElements);
    printf("✅ Test PASSED - All %d elements computed correctly!\n", numElements);
    
    // Print sample of result (first 4x4 block)
    printf("\nSample of result matrix (top-left 4x4):\n");
    for (int i = 0; i < 4 && i < height; i++)
    {
        for (int j = 0; j < 4 && j < width; j++)
        {
            printf("%6.0f ", h_C[i * width + j]);
        }
        printf("\n");
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Clean up CUDA events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("\nDone\n");
    return 0;
}