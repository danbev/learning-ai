CXX = g++
CXXFLAGS = -g -Wall -mavx2 -Wno-unused-variable

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
