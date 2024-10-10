# Makefile
CC = clang++
CFLAGS = -std=c++14 -framework Metal -framework Foundation
METAL = xcrun metal

all: simple

simple: src/simple.mm src/kernel.metal
	$(METAL) -c src/kernel.metal -o kernel.air
	xcrun metallib kernel.air -o kernel.metallib
	$(CC) $(CFLAGS) src/simple.mm -o simple

clean:
	rm -f simple *.air *.metallib
