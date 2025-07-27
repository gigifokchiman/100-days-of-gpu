NVCC = nvcc
NVCCFLAGS = -O3
BUILD_DIR = build

TARGETS = $(BUILD_DIR)/vector_add $(BUILD_DIR)/vector_add_simple $(BUILD_DIR)/vector_add_modern $(BUILD_DIR)/vector_add_modern_cpp17

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/vector_add: vector_add.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD_DIR)/vector_add_simple: vector_add_simple.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD_DIR)/vector_add_modern: vector_add_modern.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++14 -o $@ $<

$(BUILD_DIR)/vector_add_modern_cpp17: vector_add_modern_cpp17.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -o $@ $<

run: $(BUILD_DIR)/vector_add
	./$(BUILD_DIR)/vector_add

run-simple: $(BUILD_DIR)/vector_add_simple
	./$(BUILD_DIR)/vector_add_simple

run-modern: $(BUILD_DIR)/vector_add_modern
	./$(BUILD_DIR)/vector_add_modern

run-modern-cpp17: $(BUILD_DIR)/vector_add_modern_cpp17
	./$(BUILD_DIR)/vector_add_modern_cpp17

run-all: $(TARGETS)
	@echo "=== Simple Version =="
	@./$(BUILD_DIR)/vector_add_simple
	@echo
	@echo "=== Full Version ==="
	@./$(BUILD_DIR)/vector_add
	@echo
	@echo "=== Modern C++14 ==="
	@./$(BUILD_DIR)/vector_add_modern
	@echo
	@echo "=== Modern C++17 ==="
	@./$(BUILD_DIR)/vector_add_modern_cpp17

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run run-simple run-modern run-modern-cpp17 run-all clean