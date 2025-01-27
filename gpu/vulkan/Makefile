#CFLAGS = -std=c++17 -O2
LDFLAGS = -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi
CXXFLAGS += -I/usr/include/glm

VULKAN_SDK_PATH = /home/danbev/work/ai/vulkan/1.4.304.0/x86_64

DXC = $(VULKAN_SDK_PATH)/bin/dxc

simple: src/simple.cpp
	g++ $(CXXFLAGS) -o $@ $< $(LDFLAGS)

simple-shader: src/simple.glsl
	glslc -fshader-stage=compute $< -o $@.spv

shader: src/shader.frag
	#glslc -fshader-stage=compute --target-env=vulkan1.3 -O $< -o $@.spv
	glslc $< -o $@.spv

list-dev: src/list-dev.c
	gcc $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: ctags
ctags: 
	ctags -R  ~/work/ai/vulkan/1.3.283.0/x86_64/include/ .
