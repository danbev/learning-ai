CC = gcc
CXX = g++
CFLAGS = -g -Wall -Iggml/include -Iggml/src/ -Wno-unused-variable
LDFLAGS = -Lggml/build/src/

OS := $(shell uname -s)

ifeq ($(OS),Linux)
CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF
else ifeq ($(OS),Darwin)
CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_BLAS=OFF
endif

CUDA_DIR = /usr/local/cuda-12.2/lib64
CUDA_FLAGS :=

VULKAN_FLAGS :=

SRCDIR = src
BINDIR = bin

SOURCES_C := $(wildcard $(SRCDIR)/*.c)
SOURCES_CPP := $(wildcard $(SRCDIR)/*.cpp)
TARGETS_C := $(patsubst $(SRCDIR)/%.c, %, $(SOURCES_C))
TARGETS_CPP := $(patsubst $(SRCDIR)/%.cpp, %, $(SOURCES_CPP))

ifeq ($(OS),Linux)
CFLAGS += -fopenmp
WHOLE_ARCHIVE = -Wl,--whole-archive ggml/build/src/libggml.a -Wl,--no-whole-archive 
else ifeq ($(OS),Darwin)
CFLAGS += -framework Metal -framework Foundation -framework MetalKit -framework Accelerate
WHOLE_ARCHIVE = -Wl,-force_load,ggml/build/src/libggml.a
endif

.PHONY: all clean

all: $(TARGETS_C)

ggml-quants: CFLAGS += -Iggml/src -Iggml/include/ggml

$(TARGETS_C): % : $(SRCDIR)/%.c | bindir
	$(CC) $(CFLAGS) $(LDFLAGS)  -o ${BINDIR}/$@ $< \
	$(WHOLE_ARCHIVE) \
	-lm $(CUDA_FLAGS) $(VULKAN_FLAGS)

$(TARGETS_CPP): % : $(SRCDIR)/%.cpp | bindir
	$(CXX) $(CFLAGS) $(LDFLAGS)  -o ${BINDIR}/$@ $< \
	$(WHOLE_ARCHIVE) \
	-lm $(CUDA_FLAGS) $(VULKAN_FLAGS)

.PHONY: pre-quants
pre-quants: CFLAGS += -Iggml/src -Iggml/include/ggml
pre-quants:
	$(CC) -E $(CFLAGS) $(LDFLAGS) -o ${BINDIR}/quants.pre $(SRCDIR)/quants.c

.PHONY: pre-ggml.c
pre-ggml.c: CFLAGS += -Iggml/src -Iggml/include/ggml
pre-ggml.c:
	$(CC) -E $(CFLAGS) $(LDFLAGS) -o ${BINDIR}/ggml.c.pre ggml/src/ggml.c

#.PHONY: backend-cuda
backend-cuda: CUDA_FLAGS := -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
backend-cuda: CC = g++
backend-cuda: backend

backend-vulkan: VULKAN_FLAGS := -Wl,--whole-archive -lvulkan -Wl,--no-whole-archive
backend-vulkan: CC = g++
backend-vulkan: backend

.PHONY: run-backend-vulkan
run-backend-vulkan: backend-vulkan
	env GGML_VULKAN_DEVICE=1 ./bin/backend

bindir: bin

bin: 
	@mkdir -p $(BINDIR)

clean:
	${RM} -rf $(BINDIR)

.PHONY: ggml-init
ggml-init:
	cd ggml && \
	rm -rf build && mkdir build && cd build && \
	cmake ${CMAKE_FLAGS} .. && make -j8 ggml

.PHONY: ggml-init-cuda
ggml-init-cuda:
	cd ggml && \
	rm -rf build && mkdir build && cd build && \
	cmake ${CMAKE_FLAGS} -DGGML_CUDA=ON .. && make -j8 ggml

# This target does not work at the moment but should once
# https://github.com/ggerganov/ggml/pull/730 is merged.
ggml-init-vulkan:
	cd ggml && \
	rm -rf build && mkdir build && cd build && \
	cmake ${CMAKE_FLAGS} -DGGML_VULKAN=ON .. && make -j8 ggml

.PHONY: ggml-show-build-settings
ggml-show-build-settings:
	cmake -L ggml/build

.PHONY: ggml
ggml:
	cd ggml/build && make -j8

show-add-dot:
	dot -Tpng add.dot -o add.dot.png && xdg-open add.dot.png

show-mul-dot:
	dot -Tpng mul.dot -o mul.dot.png && xdg-open mul.dot.png

show-rows-dot:
	dot -Tpng get-rows.dot -o get-rows.dot.png && xdg-open get-rows.dot.png

update-ggml:
	git submodule update --remote --merge ggml

clean-ggml:
	${RM} -rf ggml/build
