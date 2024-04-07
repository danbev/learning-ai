CC = gcc
CFLAGS = -g -Wall -Iggml/include -Wno-unused-variable
LDFLAGS = -Lggml/build/src/
CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF

CUDA_DIR = /usr/local/cuda-12.2/lib64
CUDA_FLAGS :=

SRCDIR = src
BINDIR = bin

SOURCES := $(wildcard $(SRCDIR)/*.c)
TARGETS := $(patsubst $(SRCDIR)/%.c, %, $(SOURCES))

.PHONY: all clean

all: $(TARGETS)

$(TARGETS): % : $(SRCDIR)/%.c | bindir
	$(CC) $(CFLAGS) $(LDFLAGS)  -o ${BINDIR}/$@ $< \
	-Wl,--whole-archive ggml/build/src/libggml.a -Wl,--no-whole-archive \
	-lm $(CUDA_FLAGS)

#.PHONY: backend-cuda
backend-cuda: CUDA_FLAGS := -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
backend-cuda: CC = g++
backend-cuda: backend

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

.PHONY: ggml-init-vulkan
ggml-init-vulkan:
	cd ggml && \
	rm -rf build && mkdir build && cd build && \
	cmake ${CMAKE_FLAGS} .. && make -j8 C_FLAGS="-DGGML_USE_VULKAN" ggml

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
