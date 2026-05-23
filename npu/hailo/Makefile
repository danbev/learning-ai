HAILO_DIR=../hailort
RT_DIR=${HAILO_DIR}/hailort/libhailort/
RT_INCLUDE_DIR=${RT_DIR}/include
RT_BUILD_LIB_DIR = $(HAILO_DIR)/build/hailort/libhailort/src

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I$(RT_INCLUDE_DIR)

LDFLAGS = -lhailort
LDFLAGS += -L$(RT_BUILD_LIB_DIR)

version: src/version.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run-version:
	env LD_LIBRARY_PATH=$(RT_BUILD_LIB_DIR) ./version

simple: src/simple.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

inspect-hef:
	hailortcli parse-hef hefs/shortcut_net.hef

qwen: src/qwen.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

qwen-prefill: src/qwen-prefill.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

qwen-hef:
	hailortcli parse-hef hefs/qwen2.5.hef

.PHONY: ctags
ctags:
	@ctags -R /usr/include/hailo src/

.PHONY: clean
clean: 
	@${RM} simple
