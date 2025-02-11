LLGUIDANCE_DIR := /home/danbev/work/ai/llguidance
LLGUIDANCE_BUILD_DIR := $(LLGUIDANCE_DIR)/target/release

CXXFLAGS := -g -Wall -std=c++20 -I$(LLGUIDANCE_DIR)/parser

example: src/example.cpp
	$(CXX) $(CXXFLAGS) -o example src/example.cpp -L$(LLGUIDANCE_BUILD_DIR)/ -lllguidance -Wl,-rpath,$(LLGUIDANCE_BUILD_DIR)

run: example
	LD_LIBRARY_PATH=$(LLGUIDANCE_BUILD_DIR) ./example schema.json sample.json

debug: example
	LD_LIBRARY_PATH=$(LLGUIDANCE_BUILD_DIR) gdb --args ./example schema.json sample.json

ctags:
	ctags -R . $(LLGUIDANCE_DIR)/parser

clean:
	${RM} example

