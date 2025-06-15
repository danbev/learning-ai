CXX = g++
CXXFLAGS = -g -Wall -mavx2 -mavx512vnni -mavx512f -mavx512dq -mavx512vl -mavx512bw -Wno-unused-variable
CXXFLAGS = -g -Wall -mavx2 -mavx512vnni -mavx512f -mavx512dq -mavx512vl -mavx512bw -Wno-unused-variable

SRCDIR = src
BINDIR = bin

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
TARGETS := $(patsubst $(SRCDIR)/%.cpp, %, $(SOURCES))

all: $(TARGETS)

$(TARGETS): % : $(SRCDIR)/%.cpp | bindir
	$(CXX) $(CXXFLAGS) -o ${BINDIR}/$@ $<

bindir: bin
bin: 
	@mkdir -p $(BINDIR)

add: CXXFLAGS = -g -Wall -mavx2
bin/add: | bindir
	$(CXX) -march=native -v -o $@ src/add.cpp

bin/avx: src/avx.cpp | bindir
	$(CXX) -mavx -v -o $@ src/avx.cpp

bin/ssse3: src/ssse3.c | bindir
	$(CC) -mno-ssse3 -mavx -o $@ $<

saxpy: CXXFLAGS = -g -Wall -fopenmp -lgomp

# Fused Multiply and Add
.PHONY: lscpu-fma
lscpu-fma: FLAG = fma
lscpu-fma: lscpu

.PHONY: lscpu-avx
lscpu-avx: FLAG = avx
lscpu-avx: lscpu

lscpu:
	@lscpu | grep Flags | tr ' ' '\n' | grep $(FLAG)

.PHONY: clean
clean:
	${RM} -r ${BINDIR}
