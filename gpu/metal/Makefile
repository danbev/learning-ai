# Makefile
CC = clang++
CFLAGS = -std=c++14 -framework Metal -framework Foundation

all: simple

simple: src/simple.mm src/kernel.metal
	xcrun metal -c src/kernel.metal -o - | xcrun metallib - -o kernel.metallib
	$(CC) $(CFLAGS) src/simple.mm -o simple

clean:
	rm -f simple *.air *.metallib
