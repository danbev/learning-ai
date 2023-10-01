CC = gcc
CFLAGS = -g -Wall -Iggml/include
LDFLAGS = -Lggml/build/src/

ggml:
	cd ggml && rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make

all: $(TARGET)

main: src/main.c | ggml
	$(CC) $(CFLAGS) $^ -o main $(LDFLAGS) -lggml -lm

clean:
	${RM} -f $(TARGET)

