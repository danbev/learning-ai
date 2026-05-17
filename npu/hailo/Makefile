HAILO_DIR=../hailort
RT_DIR=${HAILO_DIR}/hailort/libhailort/
RT_INCLUDE_DIR=${RT_DIR}/include
RT_BUILD_LIB_DIR = $(HAILO_DIR)/build/hailort/libhailort/src

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I$(RT_INCLUDE_DIR)

LDFLAGS = -lhailort
LDFLAGS += -L$(RT_BUILD_LIB_DIR)

simple: src/simple.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run-simple:
	env LD_LIBRARY_PATH=$(RT_BUILD_LIB_DIR) ./simple

.PHONY: clean
clean: 
	@${RM} simple
