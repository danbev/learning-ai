CC = g++
SRCDIR = src
BINDIR = bin

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
TARGETS := $(patsubst $(SRCDIR)/%.cpp, %, $(SOURCES))

CXXFLAGS = -std=c++11 -Wall -Wextra -Wpedantic -Wnoexcept -fexceptions -Wnoexcept-type

.PHONY: all clean

all: $(TARGETS)

$(TARGETS): % : $(SRCDIR)/%.cpp | bindir
	$(CC) $(CXXFLAGS) -g -o ${BINDIR}/$@ $< $(LDFLAGS)

bindir: bin

bin: 
	@mkdir -p $(BINDIR)

clean:
	${RM} -rf $(BINDIR)
