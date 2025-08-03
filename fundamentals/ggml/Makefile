DEFCC  := cc
DEFCXX := c++

# Use default compilers if CC or CXX are not explicitly set
ifeq ($(origin CC),default)
CC  := $(DEFCC)
endif
ifeq ($(origin CXX),default)
CXX := $(DEFCXX)
endif

CFLAGS = -g -Wall -Iggml/include -Iggml/src/ -Wno-unused-variable -fPIC
CXXFLAGS += -g -Wall -Iggml/include -Iggml/src/ -Wno-unused-variable -std=c++11 -fPIC 
LDFLAGS = -Lggml/build/src/

OS := $(shell uname -s)

ifeq ($(OS),Linux)
CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF
else ifeq ($(OS),Darwin)
CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_BLAS=OFF
endif

CUDA_DIR = /usr/local/cuda-12.6/lib64
CUDA_FLAGS :=

VULKAN_FLAGS :=

SRCDIR = src
BINDIR = bin

SOURCES_CPP := $(wildcard $(SRCDIR)/*.cpp)
TARGETS_CPP := $(patsubst $(SRCDIR)/%.cpp, %, $(SOURCES_CPP))

ifeq ($(OS),Linux)
CFLAGS += -fopenmp
CXXFLAGS += -fopenmp
WHOLE_ARCHIVE  = -Wl,--whole-archive ggml/build/src/libggml.a -Wl,--no-whole-archive 
WHOLE_ARCHIVE += -Wl,--whole-archive ggml/build/src/libggml-base.a -Wl,--no-whole-archive 
WHOLE_ARCHIVE += -Wl,--whole-archive ggml/build/src/libggml-cpu.a -Wl,--no-whole-archive 
#WHOLE_ARCHIVE += -Wl,--whole-archive ggml/build/src/ggml-cuda/libggml-cuda.a -Wl,--no-whole-archive 
else ifeq ($(OS),Darwin)
CFLAGS += -framework Metal -framework Foundation -framework MetalKit -framework Accelerate
CXXFLAGS += -framework Metal -framework Foundation -framework MetalKit -framework Accelerate
WHOLE_ARCHIVE  = -Wl,-force_load,ggml/build/src/libggml.a
WHOLE_ARCHIVE += -Wl,-force_load,ggml/build/src/libggml-base.a
WHOLE_ARCHIVE += -Wl,-force_load,ggml/build/src/libggml-cpu.a
WHOLE_ARCHIVE += -Wl,-force_load,ggml/build/src/ggml-metal/libggml-metal.a
#WHOLE_ARCHIVE += -Wl,-force_load,ggml/build/src/libggml-cuda.a
endif

.PHONY: all clean

all: $(TARGETS_CPP)

ggml-quants: CFLAGS += -Iggml/src -Iggml/include/ggml

$(TARGETS_CPP): % : $(SRCDIR)/%.cpp | bindir
	$(CXX) $(CXXFLAGS) $(LDFLAGS)  -o ${BINDIR}/$@ $< \
	$(WHOLE_ARCHIVE) \
	-lm $(CUDA_FLAGS) $(VULKAN_FLAGS)

.PHONY: pre-quants
pre-quants: CFLAGS += -Iggml/src -Iggml/include/ggml
pre-quants:
	$(CXX) -E $(CFLAGS) $(LDFLAGS) -o ${BINDIR}/quants.pre $(SRCDIR)/quants.c

.PHONY: pre-ggml.c
pre-ggml.c: CFLAGS += -Iggml/src -Iggml/include/ggml
pre-ggml.c:
	$(CXX) -E $(CFLAGS) $(LDFLAGS) -o ${BINDIR}/ggml.c.pre ggml/src/ggml.c

#.PHONY: backend-cuda
backend-cuda: CUDA_FLAGS := -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
backend-cuda: CC = $(CXX)
backend-cuda: backend

backend-vulkan: VULKAN_FLAGS := -Wl,--whole-archive -lvulkan -Wl,--no-whole-archive
backend-vulkan: CC = $(CXX)
backend-vulkan: backend

#.PHONY: conv2d-sched-cuda
conv2d-sched-cuda: CUDA_FLAGS := -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
conv2d-sched-cuda: CC = $(CXX)
conv2d-sched-cuda: conv2d-sched

#.PHONY: sched-issue-cuda
sched-issue-cuda: CUDA_FLAGS := -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
sched-issue-cuda: CC = $(CXX)
sched-issue-cuda: sched-issue

#.PHONY: run-sched-issue
run-sched-issue: sched-issue-cuda
	env GGML_SCHED_DEBUG=2 ./bin/sched-issue

#.PHONY: debug-sched-issue
debug-sched-issue: sched-issue-cuda
	env GGML_SCHED_DEBUG=2 gdb --args ./bin/sched-issue

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

.PHONY: arch-macro
arch-macro: CFLAGS += -march=native
