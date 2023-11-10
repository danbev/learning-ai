CC = gcc
CFLAGS = -g -Wall -Iggml/include
LDFLAGS = -Lggml/build/src/

.PHONY: ggml-init
ggml-init:
	cd ggml && rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make

ggml: ggml/build
	cd ggml/build && make

all: $(TARGET)

main: src/main.c | ggml
	$(CC) $(CFLAGS) $^ -o main $(LDFLAGS) -lggml -lm

.PHONY: clean
clean:
	${RM} -f $(TARGET)
	${RM} -rf ggml/build

