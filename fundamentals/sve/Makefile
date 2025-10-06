CXX = aarch64-linux-gnu-g++
CXXFLAGS = -march=armv8-a+sve -g -static

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

run-vad: vadd-sve
	qemu-aarch64 -cpu max,sve256=on ${BINDIR}/$<

run-vec-scale: vec_scale
	qemu-aarch64 -cpu max,sve256=on ${BINDIR}/$<

.PHONY: clean
clean:
	${RM} -r ${BINDIR}
