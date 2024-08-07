#CFLAGS = -std=c++17 -O2
LDFLAGS = -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi

VULKAN_SDK_PATH = /home/danbev/work/ai/vulkan/1.3.283.0/x86_64
DXC = $(VULKAN_SDK_PATH)/bin/dxc

helloworld: src/helloworld.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)


shader: src/shader.frag
	#glslc -fshader-stage=compute --target-env=vulkan1.3 -O $< -o $@.spv
	glslc $< -o $@.spv

list-dev: src/list-dev.c
	gcc $(CFLAGS) -o $@ $< $(LDFLAGS)
