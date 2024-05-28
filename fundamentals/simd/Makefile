CXX = g++
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

bin/add: | bindir
	$(CXX) -march=native -v -o $@ src/add.cpp

bin/avx: src/avx.cpp | bindir
	$(CXX) -mavx -v -o $@ src/avx.cpp

# Fused Multiply and Add
.PHONY: lscpu-fma
lscpu-fma: FLAG = fma
lscpu-fma: lscpu

.PHONY: lscpu-avx
lscpu-avx: FLAG = avx
lscpu-avx: lscpu

lscpu:
	@lscpu | grep Flags | tr ' ' '\n' | grep $(FLAG)
