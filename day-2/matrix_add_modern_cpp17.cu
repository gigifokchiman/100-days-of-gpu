#include <iostream>
#include <memory>
#include <vector>
#include <optional>
#include <string_view>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

// ========== C++17 IMPROVEMENT: Template with [[nodiscard]] ==========
template<typename T>
class CudaMatrix {
private:
    T* d_ptr = nullptr;
    size_t width = 0;
    size_t height = 0;

public:
    [[nodiscard]] explicit CudaMatrix(size_t w, size_t h) : width(w), height(h) {
        auto result = cudaMalloc(&d_ptr, w * h * sizeof(T));
        if (result != cudaSuccess) {
            throw std::runtime_error("CUDA allocation failed: " + 
                                   std::string(cudaGetErrorString(result)));
        }
    }
    
    ~CudaMatrix() noexcept {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }
    
    // Delete copy operations
    CudaMatrix(const CudaMatrix&) = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;
    
    // Move operations with noexcept
    CudaMatrix(CudaMatrix&& other) noexcept 
        : d_ptr(other.d_ptr), width(other.width), height(other.height) {
        other.d_ptr = nullptr;
        other.width = 0;
        other.height = 0;
    }
    
    CudaMatrix& operator=(CudaMatrix&& other) noexcept {
        if (this != &other) {
            cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            width = other.width;
            height = other.height;
            other.d_ptr = nullptr;
            other.width = 0;
            other.height = 0;
        }
        return *this;
    }
    
    [[nodiscard]] T* get() noexcept { return d_ptr; }
    [[nodiscard]] size_t getWidth() const noexcept { return width; }
    [[nodiscard]] size_t getHeight() const noexcept { return height; }
    [[nodiscard]] size_t size() const noexcept { return width * height; }
    
    // ========== C++17 FEATURE: std::optional for Error Handling ==========
    [[nodiscard]] std::optional<std::string> copyFrom(const T* host_ptr) noexcept {
        auto result = cudaMemcpy(d_ptr, host_ptr, size() * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            return std::string("Copy to device failed: ") + cudaGetErrorString(result);
        }
        return std::nullopt;  // Success
    }
    
    [[nodiscard]] std::optional<std::string> copyTo(T* host_ptr) const noexcept {
        auto result = cudaMemcpy(host_ptr, d_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            return std::string("Copy from device failed: ") + cudaGetErrorString(result);
        }
        return std::nullopt;  // Success
    }
};

// Matrix addition kernel
__global__ void matrixAddKernel(const float* a, const float* b, float* c, int width, int height) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

// ========== C++17 FEATURE: Template with auto and optional ==========
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

// ========== C++17 FEATURE: Structured Bindings for Grid Config ==========
struct GridConfig {
    dim3 blocks;
    dim3 threads;
    
    [[nodiscard]] static auto calculate2D(int width, int height, int blockWidth, int blockHeight) noexcept {
        dim3 threads(blockWidth, blockHeight);
        dim3 blocks(
            (width + blockWidth - 1) / blockWidth,
            (height + blockHeight - 1) / blockHeight
        );
        return GridConfig{blocks, threads};
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
    constexpr auto WIDTH = 1024;
    constexpr auto HEIGHT = 1024;
    constexpr auto N = WIDTH * HEIGHT;
    
    std::cout << "[Matrix addition of " << WIDTH << "x" << HEIGHT << " matrices]" << std::endl;
    
    try {
        // ========== C++17 IMPROVEMENT: Class Template Argument Deduction ==========
        std::vector h_a(N, 0.0f);  // Deduces std::vector<float>
        std::vector h_b(N, 0.0f);
        std::vector h_c(N, 0.0f);
        
        // Initialize matrices
        for (size_t i = 0; i < N; ++i) {
            h_a[i] = static_cast<float>(i);
            h_b[i] = static_cast<float>(i * 2);
        }
        
        // ========== TIMING SETUP ==========
        CudaTimer cuda_timer;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        // ========== C++17 FEATURE: Automatic Scope Management ==========
        {
            auto d_a = CudaMatrix<float>(WIDTH, HEIGHT);
            auto d_b = CudaMatrix<float>(WIDTH, HEIGHT);
            auto d_c = CudaMatrix<float>(WIDTH, HEIGHT);
            
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
            auto [blocks, threads] = GridConfig::calculate2D(WIDTH, HEIGHT, 16, 16);
            
            std::cout << "CUDA kernel launch with " << blocks.x << "x" << blocks.y 
                      << " blocks of " << threads.x << "x" << threads.y 
                      << " threads" << std::endl;
            
            // Start GPU timer (kernel only)
            cuda_timer.start();
            
            // Launch kernel with modern error handling
            if (auto error = launchKernel(matrixAddKernel, blocks, threads, 
                                        d_a.get(), d_b.get(), d_c.get(), WIDTH, HEIGHT); error) {
                std::cerr << "Kernel error: " << *error << std::endl;
                return 1;
            }
            
            // Stop GPU timer
            float gpu_time = cuda_timer.stop();
            
            // Copy results back
            if (auto error = d_c.copyTo(h_c.data()); error) {
                std::cerr << "Error: " << *error << std::endl;
                return 1;
            }
            
            // Calculate CPU time
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
        
        // Print sample of result
        std::cout << "\nSample of result matrix (top-left 4x4):" << std::endl;
        for (int i = 0; i < 4 && i < HEIGHT; i++) {
            for (int j = 0; j < 4 && j < WIDTH; j++) {
                std::cout << std::fixed << std::setw(6) << h_c[i * WIDTH + j] << " ";
            }
            std::cout << std::endl;
        }
        
        if (auto error_index = verify_results(); error_index) {
            std::cerr << "\nVerification failed at element " << *error_index << std::endl;
            return 1;
        }
        
        std::cout << "\n✅ Test PASSED - All " << N << " elements computed correctly!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}