# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_70

# C++ standard flags for modern versions
CPP14_FLAGS = -std=c++14
CPP17_FLAGS = -std=c++17

# Target executables
TARGETS = build/matrix_add build/matrix_add_1d build/matrix_add_modern_cpp17

# Default target
all: $(TARGETS)

# Full version
build/matrix_add: matrix_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# 1D vs 2D block comparison
build/matrix_add_1d: matrix_add_1d.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Modern C++17 version
build/matrix_add_modern_cpp17: matrix_add_modern_cpp17.cu
	$(NVCC) $(NVCC_FLAGS) $(CPP17_FLAGS) -o $@ $<

# Run all versions
run-all: $(TARGETS)
	@echo "=== Full Version (1024x1024 matrix) ==="
	@./build/matrix_add
	@echo ""
	@echo "=== 1D vs 2D Block Comparison ==="
	@./build/matrix_add_1d
	@echo ""
	@echo "=== Modern C++17 Version ==="
	@./build/matrix_add_modern_cpp17


# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Create build directory if it doesn't exist
$(shell mkdir -p build)

.PHONY: all run-all clean