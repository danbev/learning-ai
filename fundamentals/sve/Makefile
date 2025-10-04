CXX = g++
CXXFLAGS = -march=armv8-a+sve -g

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

.PHONY: clean
clean:
	${RM} -r ${BINDIR}
