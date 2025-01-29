DAWN_DIR = $(HOME)/work/webgpu/dawn
DAWN_BUILD_DIR = ${DAWN_DIR}/out/Debug

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
CXXFLAGS = -std=c++17 -Wall -Wextra -g

DAWN_LIBS = $(DAWN_BUILD_DIR)/src/dawn/native/libdawn_native.a \
            $(DAWN_BUILD_DIR)/src/dawn/libdawn_proc.a \
            $(DAWN_BUILD_DIR)/src/dawn/common/libdawn_common.a

INCLUDES = -I$(DAWN_BUILD_DIR)/gen/include -I$(DAWN_DIR)/include

SRC_DIR = src
BUILD_DIR = bin
TARGET = $(BUILD_DIR)/matrix-mul

SRC = $(SRC_DIR)/matrix-mul.cpp
OBJ = $(BUILD_DIR)/matrix-mul.o

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJ)
	$(CXX) -L/home/danbev/work/webgpu/dawn/out/Debug/src/dawn/native/  $(OBJ) -Wl,--whole-archive -lwebgpu_dawn -Wl,--no-whole-archive $(DAWN_LIBS) -o $(TARGET)

$(BUILD_DIR)/%.o: $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

bin/list-adapters: src/list-adapters.cpp
	$(CXX) -L${DAWN_BUILD_DIR}/src/dawn/native \
	-Wl,--whole-archive -ldawn_native -lwebgpu_dawn -ldawn_platform -ldawn_common -ldawn_proc -Wl,--no-whole-archive \
	-L/home/danbev/work/webgpu/dawn/out/Debug/src/dawn/common \
	-L/home/danbev/work/webgpu/dawn/out/Debug/src/dawn/platform \
	-L/home/danbev/work/webgpu/dawn/out/Debug/src/dawn/ \
	-ldawn_proc -ldawn_common -ldawn_platform -lwebgpu_dawn -ldawn_native \
	$(CXXFLAGS) $(INCLUDES) \
	$< -o $@

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run help
