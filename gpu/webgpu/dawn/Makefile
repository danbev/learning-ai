DAWN_DIR = $(HOME)/work/webgpu/dawn
DAWN_BUILD_DIR = ${DAWN_DIR}/install/Release

configure-dawn:
	cd ${DAWN_DIR} && cmake -B build -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DDAWN_BUILD_SAMPLES=OFF \
    -DDAWN_BUILD_TESTS=OFF \
    -DDAWN_ENABLE_D3D12=OFF \
    -DDAWN_ENABLE_METAL=OFF \
    -DDAWN_ENABLE_VULKAN=ON

build-dawn: configure
	cd ${DAWN_DIR} && cmake --build build

.PHONY: configure build

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -g

INCLUDES = -I$(DAWN_BUILD_DIR)/include

SRC_DIR = src
BUILD_DIR = bin
TARGET = $(BUILD_DIR)/matrix-mul

SRC = $(SRC_DIR)/matrix-mul.cpp
OBJ = $(BUILD_DIR)/matrix-mul.o

all: $(TARGET) | $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJ) | $(BUILD_DIR)
	$(CXX) -L${DAWN_BUILD_DIR}/lib  $(OBJ) -lwebgpu_dawn -o $(TARGET)

$(BUILD_DIR)/%.o: $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

list-adapters: src/list-adapters.cpp | $(BUILD_DIR)
	$(CXX) -L${DAWN_BUILD_DIR}/lib -ldl \
		$(CXXFLAGS) $(INCLUDES) \
		$< -o ${BUILD_DIR}/$@ -lwebgpu_dawn

run-list-adapters: list-adapters
	LD_LIBRARY_PATH=${DAWN_BUILD_DIR}/lib DYLD_LIBRARY_PATH=${DAWN_BUILD_DIR}/lib \
		VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
		${BUILD_DIR}/list-adapters

features: src/features.cpp | $(BUILD_DIR)
	$(CXX) -L${DAWN_BUILD_DIR}/lib -ldl \
		$(CXXFLAGS) $(INCLUDES) \
		$< -o ${BUILD_DIR}/$@ -lwebgpu_dawn

run-features: features
	LD_LIBRARY_PATH=${DAWN_BUILD_DIR}/lib DYLD_LIBRARY_PATH=${DAWN_BUILD_DIR}/lib \
		VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
		${BUILD_DIR}/features

vector-addition: src/vector-addition.cpp | $(BUILD_DIR)
	$(CXX) -L${DAWN_BUILD_DIR}/lib -ldl \
		$(CXXFLAGS) $(INCLUDES) \
		$< -o ${BUILD_DIR}/$@ -lwebgpu_dawn

run-vector-addition: vector-addition
	LD_LIBRARY_PATH=${DAWN_BUILD_DIR}/lib DYLD_LIBRARY_PATH=${DAWN_BUILD_DIR}/lib \
		VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
		${BUILD_DIR}/vector-addition

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run help
