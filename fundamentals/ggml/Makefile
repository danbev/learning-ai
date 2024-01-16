CC = gcc
CFLAGS = -g -Wall -Iggml/include
LDFLAGS = -Lggml/build/src/

.PHONY: ggml-init
ggml-init:
	cd ggml && rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make

.PHONY: ggml
ggml:
	cd ggml/build && make

all: $(TARGET)

graph: src/graph.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

tensor: src/tensor.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

matrix-mul: src/matrix-mul.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

show-add-dot:
	dot -Tpng add.dot -o add.dot.png && xdg-open add.dot.png

update-ggml:
	git submodule update --remote --merge ggml

.PHONY: clean
clean:
	${RM} -f graph tensor


clean-ggml:
	${RM} -rf ggml/build
