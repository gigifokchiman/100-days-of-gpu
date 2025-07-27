NVCC = nvcc
NVCCFLAGS = -O3
BUILD_DIR = build

TARGETS = $(BUILD_DIR)/vector_add $(BUILD_DIR)/vector_add_simple $(BUILD_DIR)/vector_add_modern

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/vector_add: vector_add.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD_DIR)/vector_add_simple: vector_add_simple.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD_DIR)/vector_add_modern: vector_add_modern.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++14 -o $@ $<

run: $(BUILD_DIR)/vector_add
	./$(BUILD_DIR)/vector_add

run-simple: $(BUILD_DIR)/vector_add_simple
	./$(BUILD_DIR)/vector_add_simple

run-modern: $(BUILD_DIR)/vector_add_modern
	./$(BUILD_DIR)/vector_add_modern

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run run-simple run-modern clean