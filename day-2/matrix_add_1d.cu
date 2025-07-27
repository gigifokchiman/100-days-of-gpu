#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

// ========== 1D BLOCK APPROACH: Treats matrix as flattened array ==========
// Each thread processes one element using linear indexing
__global__ void matrixAdd1D(float *a, float *b, float *c, int total_elements, int width)
{
    // Calculate linear thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Check bounds
    if (idx < total_elements)
    {
        // ========== KEY DIFFERENCE: Need to calculate row/col from linear index ==========
        // This requires expensive integer division and modulo operations
        int row = idx / width;    // Integer division (slow!)
        int col = idx % width;    // Modulo operation (slow!)
        
        // Perform matrix addition
        c[idx] = a[idx] + b[idx];
        
        // Note: We calculated row/col but didn't actually need them for addition!
        // But for more complex operations (like transpose), these would be essential
    }
}

// ========== 2D BLOCK APPROACH: Natural matrix indexing ==========
__global__ void matrixAdd2D(float *a, float *b, float *c, int width, int height)
{
    // Direct 2D indexing - no division or modulo needed!
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Check bounds
    if (col < width && row < height)
    {
        int idx = row * width + col;  // Simple multiplication and addition
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    // Matrix dimensions
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int TOTAL_ELEMENTS = WIDTH * HEIGHT;
    const int SIZE = TOTAL_ELEMENTS * sizeof(float);
    
    printf("=== Matrix Addition: 1D vs 2D Block Comparison ===\n");
    printf("Matrix size: %dx%d (%d elements)\n\n", WIDTH, HEIGHT, TOTAL_ELEMENTS);
    
    // Allocate host memory
    float *h_a = (float*)malloc(SIZE);
    float *h_b = (float*)malloc(SIZE);
    float *h_c_1d = (float*)malloc(SIZE);
    float *h_c_2d = (float*)malloc(SIZE);
    
    // Initialize matrices
    for (int i = 0; i < TOTAL_ELEMENTS; i++)
    {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c_1d, *d_c_2d;
    cudaMalloc(&d_a, SIZE);
    cudaMalloc(&d_b, SIZE);
    cudaMalloc(&d_c_1d, SIZE);
    cudaMalloc(&d_c_2d, SIZE);
    
    // Copy input data to device
    cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);
    
    // ========== CUDA EVENTS FOR PRECISE TIMING ==========
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========== TEST 1: 1D BLOCK APPROACH ==========
    printf("=== 1D Block Approach ===\n");
    
    // 1D grid configuration
    int threadsPerBlock1D = 256;
    int blocksPerGrid1D = (TOTAL_ELEMENTS + threadsPerBlock1D - 1) / threadsPerBlock1D;
    
    printf("CUDA kernel launch with %d blocks of %d threads (1D)\n", 
           blocksPerGrid1D, threadsPerBlock1D);
    printf("Total threads: %d\n", blocksPerGrid1D * threadsPerBlock1D);
    
    // Time 1D approach
    cudaEventRecord(start);
    matrixAdd1D<<<blocksPerGrid1D, threadsPerBlock1D>>>(d_a, d_b, d_c_1d, TOTAL_ELEMENTS, WIDTH);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time1D = 0;
    cudaEventElapsedTime(&time1D, start, stop);
    
    // Copy result back
    cudaMemcpy(h_c_1d, d_c_1d, SIZE, cudaMemcpyDeviceToHost);
    
    printf("GPU computation time (1D): %.3f ms\n\n", time1D);
    
    // ========== TEST 2: 2D BLOCK APPROACH ==========
    printf("=== 2D Block Approach ===\n");
    
    // 2D grid configuration
    dim3 threadsPerBlock2D(16, 16);  // 256 threads total (same as 1D)
    dim3 blocksPerGrid2D(
        (WIDTH + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
        (HEIGHT + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y
    );
    
    printf("CUDA kernel launch with %dx%d blocks of %dx%d threads (2D)\n",
           blocksPerGrid2D.x, blocksPerGrid2D.y, 
           threadsPerBlock2D.x, threadsPerBlock2D.y);
    printf("Total threads: %d\n", 
           blocksPerGrid2D.x * blocksPerGrid2D.y * threadsPerBlock2D.x * threadsPerBlock2D.y);
    
    // Time 2D approach
    cudaEventRecord(start);
    matrixAdd2D<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_a, d_b, d_c_2d, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time2D = 0;
    cudaEventElapsedTime(&time2D, start, stop);
    
    // Copy result back
    cudaMemcpy(h_c_2d, d_c_2d, SIZE, cudaMemcpyDeviceToHost);
    
    printf("GPU computation time (2D): %.3f ms\n\n", time2D);
    
    // ========== VERIFICATION ==========
    printf("=== Verification ===\n");
    bool results_match = true;
    for (int i = 0; i < TOTAL_ELEMENTS; i++)
    {
        if (h_c_1d[i] != h_c_2d[i])
        {
            printf("Mismatch at element %d: 1D=%f, 2D=%f\n", i, h_c_1d[i], h_c_2d[i]);
            results_match = false;
            break;
        }
        
        // Also verify correctness
        float expected = h_a[i] + h_b[i];
        if (h_c_1d[i] != expected)
        {
            printf("Incorrect result at element %d: got %f, expected %f\n", i, h_c_1d[i], expected);
            results_match = false;
            break;
        }
    }
    
    if (results_match)
    {
        printf("✅ Both approaches produce identical correct results!\n");
    }
    else
    {
        printf("❌ Results differ between approaches!\n");
    }
    
    // ========== PERFORMANCE COMPARISON ==========
    printf("\n=== Performance Analysis ===\n");
    printf("1D approach: %.3f ms\n", time1D);
    printf("2D approach: %.3f ms\n", time2D);
    
    if (time1D > time2D)
    {
        float speedup = time1D / time2D;
        printf("2D approach is %.2fx faster!\n", speedup);
        printf("Improvement: %.1f%% faster\n", ((time1D - time2D) / time1D) * 100);
    }
    else if (time2D > time1D)
    {
        float speedup = time2D / time1D;
        printf("1D approach is %.2fx faster!\n", speedup);
        printf("Note: This is unusual - may indicate measurement noise\n");
    }
    else
    {
        printf("Both approaches have similar performance\n");
    }
    
    // ========== MEMORY ACCESS PATTERN ANALYSIS ==========
    printf("\n=== Memory Access Pattern Analysis ===\n");
    printf("1D Block Pattern:\n");
    printf("  - Threads 0-255 process elements 0-255 (rows 0-1)\n");
    printf("  - Threads 256-511 process elements 256-511 (rows 1-2)\n");
    printf("  - Sequential access, good coalescing within warps\n");
    printf("  - But: requires division/modulo for row/col calculation\n\n");
    
    printf("2D Block Pattern:\n");
    printf("  - Thread (0,0) to (15,15) process 16x16 tile\n");
    printf("  - Threads in same warp (consecutive threadIdx.x) access consecutive columns\n");
    printf("  - Perfect coalescing: threads 0-31 in warp access A[row*width+0] to A[row*width+31]\n");
    printf("  - No division/modulo needed - direct row/col from indices\n\n");
    
    // ========== SAMPLE OUTPUT ==========
    printf("=== Sample Results (top-left 4x4) ===\n");
    printf("Input A:\n");
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("%6.0f ", h_a[i * WIDTH + j]);
        }
        printf("\n");
    }
    
    printf("\nInput B:\n");
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("%6.0f ", h_b[i * WIDTH + j]);
        }
        printf("\n");
    }
    
    printf("\nResult (A + B):\n");
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("%6.0f ", h_c_2d[i * WIDTH + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_1d);
    free(h_c_2d);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_2d);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n✅ Test completed successfully!\n");
    return 0;
}