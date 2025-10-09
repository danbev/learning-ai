OPENVINO_DIR     := deps/openvino_toolkit_ubuntu24_2025.3.0.19807.44526285f24_x86_64
OPENVINO_INCLUDE := $(OPENVINO_DIR)/runtime/include
OPENVINO_LIB     := $(OPENVINO_DIR)/runtime/lib/intel64

CXX = g++
CXXFLAGS := -std=c++17 -Wall -Wextra -g -I$(OPENVINO_INCLUDE)

LDFLAGS := -L$(OPENVINO_LIB) -Wl,-rpath,$(OPENVINO_LIB)
LIBS := -lopenvino

SRCDIR = src
BINDIR = bin

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
TARGETS := $(patsubst $(SRCDIR)/%.cpp, %, $(SOURCES))

all: $(TARGETS)

$(TARGETS): % : $(SRCDIR)/%.cpp | bindir
	$(CXX) $(CXXFLAGS) -o ${BINDIR}/$@ $< $(LDFLAGS) $(LIBS)

bindir: bin
bin:
	@mkdir -p $(BINDIR)

.PHONY: clean
clean:
	${RM} -r ${BINDIR}
