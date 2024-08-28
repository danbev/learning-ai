# Makefile
CC = clang++
CFLAGS = -std=c++14 -framework Metal -framework Foundation
METAL = xcrun metal

all: simple

simple: src/main.mm src/compute.cpp src/kernel.metal
	$(METAL) -c src/kernel.metal -o kernel.air
	xcrun metallib kernel.air -o kernel.metallib
	$(CC) $(CFLAGS) src/main.mm src/compute.cpp -o simple

clean:
	rm -f simple *.air *.metallib
