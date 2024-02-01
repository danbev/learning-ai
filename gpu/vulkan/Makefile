#CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

VULKAN_SDK_PATH = /home/danielbevenius/work/ai/vulkan/1.3.275.0/x86_64
DXC = $(VULKAN_SDK_PATH)/bin/dxc

helloworld: src/helloworld.cpp
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)


shader: src/shader.frag
	#glslc -fshader-stage=compute --target-env=vulkan1.3 -O $< -o $@.spv
	glslc $< -o $@.spv

list-dev: src/list-dev.c
	gcc $(CFLAGS) -o $@ $< $(LDFLAGS)
