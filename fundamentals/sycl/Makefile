CXX=dpcpp
CXXFLAGS=-O2 -g -std=c++17

all: main

add: src/add.cpp
	 icpx -fsycl $< -o $@ \
	 -I/opt/intel/oneapi/compiler/2024.0/include/sycl \
	 -I/opt/intel/oneapi/compiler/2024.0/include -std=c++17 \
	 -fsycl-unnamed-lambda

clean:
	${RM} -f add

