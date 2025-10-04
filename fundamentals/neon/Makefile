CXX = g++
CXXFLAGS =

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

vadd-sve: CXXFLAGS = -march=armv8-a+sve -g
bin/add: src/add-sve.cpp | bindir
	$(CXX) -march=native -v -o $@ $<

.PHONY: clean
clean:
	${RM} -r ${BINDIR}
