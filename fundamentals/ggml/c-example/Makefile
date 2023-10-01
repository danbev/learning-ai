CXX = g++
CXXFLAGS = -g -Wall -Iggml/include
LDFLAGS = -Lggml/build/src/

ggml:
	cd ggml && rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make

all: $(TARGET)

main: src/main.cpp | ggml
	$(CXX) $(CXXFLAGS) $^ -o main $(LDFLAGS) -lggml

clean:
	${RM} -f $(TARGET)

