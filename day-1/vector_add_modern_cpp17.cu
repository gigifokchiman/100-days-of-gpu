#include <iostream>
#include <memory>
#include <vector>
#include <optional>
#include <string_view>
#include <chrono>
#include <cuda_runtime.h>

// ========== C++17 IMPROVEMENT: Class Template Argument Deduction ==========
// In C++17, we can omit template arguments when they can be deduced
template<typename T>
class CudaBuffer {
private:
    T* d_ptr = nullptr;
    size_t size = 0;

public:
    // ========== C++17 FEATURE: [[nodiscard]] Attribute ==========
    // Warns if return value is ignored - prevents memory leaks
    [[nodiscard]] explicit CudaBuffer(size_t count) : size(count) {
        auto result = cudaMalloc(&d_ptr, count * sizeof(T));
        if (result != cudaSuccess) {
            throw std::runtime_error("CUDA allocation failed: " + 
                                   std::string(cudaGetErrorString(result)));
        }
    }
    
    // ========== C++17 IMPROVEMENT: Exception-Safe Destructor ==========
    ~CudaBuffer() noexcept {
        if (d_ptr) {
            cudaFree(d_ptr);  // noexcept operations in destructor
        }
    }
    
    // Delete copy operations (Rule of Five)
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // ========== C++17 FEATURE: noexcept Specification ==========
    // Move operations are noexcept, enabling optimizations
    CudaBuffer(CudaBuffer&& other) noexcept : d_ptr(other.d_ptr), size(other.size) {
        other.d_ptr = nullptr;
        other.size = 0;
    }
    
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            size = other.size;
            other.d_ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }
    
    [[nodiscard]] T* get() noexcept { return d_ptr; }
    [[nodiscard]] size_t count() const noexcept { return size; }
    
    // ========== C++17 FEATURE: std::optional for Safe Operations ==========
    [[nodiscard]] std::optional<std::string> copyFrom(const T* host_ptr) noexcept {
        auto result = cudaMemcpy(d_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            return std::string("Copy to device failed: ") + cudaGetErrorString(result);
        }
        return std::nullopt;  // Success
    }
    
    [[nodiscard]] std::optional<std::string> copyTo(T* host_ptr) const noexcept {
        auto result = cudaMemcpy(host_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            return std::string("Copy from device failed: ") + cudaGetErrorString(result);
        }
        return std::nullopt;  // Success
    }
};

// Original kernel - unchanged
__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// ========== C++17 FEATURE: if constexpr and auto Templates ==========
template<typename KernelFunc, typename... Args>
[[nodiscard]] std::optional<std::string> launchKernel(KernelFunc kernel, dim3 blocks, dim3 threads, Args... args) {
    kernel<<<blocks, threads>>>(args...);
    
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        return std::string("Kernel launch failed: ") + cudaGetErrorString(error);
    }
    
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        return std::string("Kernel execution failed: ") + cudaGetErrorString(error);
    }
    
    return std::nullopt;  // Success
}

// ========== C++17 FEATURE: Structured Bindings ==========
struct KernelConfig {
    int blocks;
    int threads;
    
    // C++17: Can be used with structured bindings
    [[nodiscard]] auto calculate(int total_elements) const noexcept {
        int calculated_blocks = (total_elements + threads - 1) / threads;
        return std::make_pair(calculated_blocks, threads);
    }
};

// ========== C++17 TIMER: Modern CUDA Event Timer ==========
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
    
    [[nodiscard]] float stop() {
        if (!is_started) {
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
    // ========== C++17 FEATURE: Inline Variables ==========
    constexpr auto N = 1'000'000;  // C++14 digit separators still work
    
    try {
        // ========== C++17 IMPROVEMENT: Class Template Argument Deduction ==========
        std::vector h_a(N, 0.0f);  // Deduces std::vector<float>
        std::vector h_b(N, 0.0f);  // No need to specify <float>
        std::vector h_c(N, 0.0f);
        
        // Initialize with C++17 style
        for (size_t i = 0; i < N; ++i) {
            h_a[i] = static_cast<float>(i);
            h_b[i] = static_cast<float>(i * 2);
        }
        
        // ========== TIMING SETUP ==========
        CudaTimer cuda_timer;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        // ========== C++17 FEATURE: Automatic Scope Management ==========
        {
            // Class template argument deduction works with our custom class too
            auto d_a = CudaBuffer<float>(N);  // Could be just CudaBuffer(N) if we had deduction guides
            auto d_b = CudaBuffer<float>(N);
            auto d_c = CudaBuffer<float>(N);
            
            // Start GPU timer just before GPU operations
            cuda_timer.start();
            
            // ========== C++17 FEATURE: std::optional Error Handling ==========
            if (auto error = d_a.copyFrom(h_a.data()); error) {
                std::cerr << "Error: " << *error << std::endl;
                return 1;
            }
            
            if (auto error = d_b.copyFrom(h_b.data()); error) {
                std::cerr << "Error: " << *error << std::endl;
                return 1;
            }
            
            // ========== C++17 FEATURE: Structured Bindings ==========
            const KernelConfig config{.blocks = 0, .threads = 256};  // C++20 designated initializers (if supported)
            auto [blocks, threads] = config.calculate(N);
            
            std::cout << "CUDA kernel launch with " << blocks << " blocks of " 
                      << threads << " threads" << std::endl;
            
            // Launch kernel with modern error handling
            if (auto error = launchKernel(addKernel, blocks, threads, 
                                        d_a.get(), d_b.get(), d_c.get(), N); error) {
                std::cerr << "Kernel error: " << *error << std::endl;
                return 1;
            }
            
            // Copy results back
            if (auto error = d_c.copyTo(h_c.data()); error) {
                std::cerr << "Error: " << *error << std::endl;
                return 1;
            }
            
            // Stop GPU timer after all GPU operations
            float gpu_time = cuda_timer.stop();
            
            // Calculate CPU time including all operations
            auto cpu_end = std::chrono::high_resolution_clock::now();
            auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
            float cpu_time_ms = cpu_duration.count() / 1000.0f;
            
            std::cout << "GPU computation time: " << gpu_time << " ms" << std::endl;
            std::cout << "Total CPU time: " << cpu_time_ms << " ms" << std::endl;
            
        } // ========== AUTOMATIC GPU MEMORY CLEANUP HERE ==========
        
        // ========== C++17 FEATURE: std::optional Verification ==========
        auto verify_results = [&]() -> std::optional<size_t> {
            for (size_t i = 0; i < N; ++i) {
                if (std::abs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
                    return i;  // Return index of first error
                }
            }
            return std::nullopt;  // Success
        };
        
        if (auto error_index = verify_results(); error_index) {
            std::cerr << "Verification failed at element " << *error_index << std::endl;
            return 1;
        }
        
        std::cout << "✅ Test PASSED - All " << N << " elements computed correctly!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}