# Makefile
CC = clang++
CFLAGS = -g -std=c++14 -framework Metal -framework Foundation

all: simple

simple: src/simple.mm src/kernel.metal
	xcrun metal -c src/kernel.metal -o - | xcrun metallib - -o kernel.metallib
	$(CC) $(CFLAGS) src/simple.mm -o simple

simple-source: src/simple-source.mm
	$(CC) $(CFLAGS) $< -o $@

cumsum: src/cumsum.mm src/cumsum.metal
	xcrun metal -c src/cumsum.metal -o - | xcrun metallib - -o cumsum.metallib
	$(CC) $(CFLAGS) src/cumsum.mm -o $@

cumsum_simd: src/cumsum_simd.mm src/cumsum_simd.metal
	xcrun metal -c src/cumsum_simd.metal -o - | xcrun metallib - -o cumsum_simd.metallib
	$(CC) $(CFLAGS) src/cumsum_simd.mm -o $@

clean:
	${RM} -f simple* *.air *.metallib
