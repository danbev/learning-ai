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

qwen-tg: src/qwen-tg.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

qwen-prefill: src/qwen-prefill.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

qwen-hef:
	@hailortcli parse-hef hefs/qwen2.5.hef

download-qwen-2.5:
	curl -L -o hefs/Qwen2-1.5B-Instruct.hef "https://dev-public.hailo.ai/v5.3.0/blob/Qwen2-1.5B-Instruct.hef"

qwen-infer: src/qwen-infer.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

extract-theta: src/extract-theta.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

extract-config: src/extract-config.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

dump-meta: src/dump-output-meta.cpp
	$(CXX) -std=c++17 $< -o $@ -lhailort

.PHONY: identify
identify:
	@hailortcli fw-control identify

.PHONY: scan
scan:
	@hailortcli scan

.PHONY: ctags
ctags:
	@ctags -R /usr/include/hailo src/

.PHONY: clean
clean: 
	@${RM} simple qwen-tg qwen-prefill qwen-infer
