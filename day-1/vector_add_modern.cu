#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ========== MODERN C++ IMPROVEMENT: RAII Pattern ==========
// Resource Acquisition Is Initialization - automatic memory management
// No manual cudaFree needed - destructor handles it automatically!
// Similar to Python's 'with' statement or C#'s 'using'
template<typename T>
class CudaBuffer {
private:
    T* d_ptr = nullptr;
    size_t size = 0;

public:
    explicit CudaBuffer(size_t count) : size(count) {
        cudaMalloc(&d_ptr, count * sizeof(T));
    }
    
    // ========== KEY IMPROVEMENT: Automatic Cleanup ==========
    // Destructor automatically called when object goes out of scope
    // No memory leaks even if exceptions are thrown!
    ~CudaBuffer() {
        if (d_ptr) cudaFree(d_ptr);
    }
    
    // ========== MODERN C++ SAFETY: Rule of Five ==========
    // Delete copy operations to prevent double-free bugs
    // Only one object can own the GPU memory at a time
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // ========== MODERN C++ EFFICIENCY: Move Semantics ==========
    // Allow efficient transfer of ownership without copying
    // Critical for returning CudaBuffer from functions
    CudaBuffer(CudaBuffer&& other) noexcept : d_ptr(other.d_ptr), size(other.size) {
        other.d_ptr = nullptr;
        other.size = 0;
    }
    
    T* get() { return d_ptr; }
    size_t count() const { return size; }
    
    // Convenient copy from host
    void copyFrom(const T* host_ptr) {
        cudaMemcpy(d_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    }
    
    // Convenient copy to host
    void copyTo(T* host_ptr) const {
        cudaMemcpy(host_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

// Kernel remains the same
__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// ========== MODERN C++ IMPROVEMENT: Exception-Safe Error Handling ==========
// Encapsulates error checking in a reusable class
// No more repetitive if-statements after every CUDA call
class CudaKernel {
public:
    static void checkError(cudaError_t error, const char* file, int line) {
        if (error != cudaSuccess) {
            std::cerr << "CUDA error at " << file << ":" << line 
                      << " - " << cudaGetErrorString(error) << std::endl;
            std::exit(1);
        }
    }
    
    template<typename KernelFunc, typename... Args>
    static void launch(KernelFunc kernel, dim3 blocks, dim3 threads, Args... args) {
        kernel<<<blocks, threads>>>(args...);
        checkError(cudaGetLastError(), __FILE__, __LINE__);
        checkError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
};

// ========== MODERN C++ CONVENIENCE: Error Checking Macro ==========
// Automatically includes file and line number for debugging
#define CUDA_CHECK(call) CudaKernel::checkError(call, __FILE__, __LINE__)

// ========== MODERN C++ TIMER: CUDA Event-based Timer ==========
// Uses CUDA events for precise GPU timing
// RAII pattern ensures events are properly created and destroyed
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    bool is_started = false;
    
public:
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event);
        is_started = true;
    }
    
    float stop() {
        if (!is_started) {
            std::cerr << "Timer not started!" << std::endl;
            return 0.0f;
        }
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        is_started = false;
        return milliseconds;
    }
};

int main() {
    // ========== MODERN C++ READABILITY: Digit Separators ==========
    constexpr int N = 1'000'000;  // C++14 feature - much easier to read than 1000000
    
    // ========== MODERN C++ IMPROVEMENT: std::vector ==========
    // No manual malloc/free needed for host memory
    // Automatic resize, bounds checking in debug mode
    // Exception safe - no leaks if constructor throws
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);
    
    // Initialize with modern C++ style
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);      // Explicit casting for clarity
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // ========== TIMING SETUP ==========
    CudaTimer cuda_timer;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // ========== CRITICAL IMPROVEMENT: Automatic GPU Memory Management ==========
    // This scope demonstrates the power of RAII:
    // - GPU memory allocated in constructors
    // - Automatically freed at closing brace '}'
    // - Works even if exceptions are thrown!
    // - No possibility of forgetting cudaFree()
    {
        CudaBuffer<float> d_a(N);  // Constructor calls cudaMalloc
        CudaBuffer<float> d_b(N);
        CudaBuffer<float> d_c(N);
        
        // Start GPU timer just before GPU operations
        cuda_timer.start();
        
        // ========== MODERN C++ API: Encapsulated Operations ==========
        // Methods hide error checking and size calculations
        d_a.copyFrom(h_a.data());  // vs cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice)
        d_b.copyFrom(h_b.data());
        
        // Launch kernel with error checking
        constexpr int threadsPerBlock = 256;
        const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        
        std::cout << "CUDA kernel launch with " << blocksPerGrid 
                  << " blocks of " << threadsPerBlock << " threads" << std::endl;
        
        CudaKernel::launch(addKernel, blocksPerGrid, threadsPerBlock, 
                          d_a.get(), d_b.get(), d_c.get(), N);
        
        // Copy result back
        d_c.copyTo(h_c.data());
        
        // Stop GPU timer after all GPU operations
        float gpu_time = cuda_timer.stop();
        
        // Calculate CPU time including all operations
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        float cpu_time_ms = cpu_duration.count() / 1000.0f;
        
        std::cout << "GPU computation time: " << gpu_time << " ms" << std::endl;
        std::cout << "Total CPU time: " << cpu_time_ms << " ms" << std::endl;
        
    } // ========== AUTOMATIC CLEANUP HERE! ==========
      // All three CudaBuffer destructors called
      // GPU memory freed without explicit cudaFree() calls
      // This prevents GPU memory leaks even in complex code paths
    
    // ========== MODERN C++ STYLE: Boolean Flag Pattern ==========
    // More idiomatic than multiple exit() calls
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-5) {
            std::cerr << "Error at element " << i << ": " 
                      << h_c[i] << " != " << expected << std::endl;
            success = false;
            break;
        }
    }
    
    std::cout << (success ? "✅ Test PASSED - All " + std::to_string(N) + " elements computed correctly!" : "Test FAILED") << std::endl;
    return success ? 0 : 1;
}