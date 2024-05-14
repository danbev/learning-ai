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
