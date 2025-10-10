CXX = g++
CXXFLAGS = -march=sapphirerapids -g -static
SDE_DIR = deps/sde-external-9.58.0-2025-06-16-lin
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

run-vnni: vnni
	${SDE_DIR}/sde64 -spr -- ${BINDIR}/$<

.PHONY: clean
clean:
	${RM} -r ${BINDIR}
