#CFLAGS = -std=c++17 -O2
LDFLAGS = -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi
CXXFLAGS += -I/usr/include/glm

VULKAN_SDK_PATH = /home/danbev/work/ai/vulkan/1.4.304.0/x86_64

#export VK_LOADER_DEBUG=all

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

simple-layer: src/simple-layer.c
	g++ -g -shared -fPIC -o libsimple_layer.so $< $(LDFLAGS)

# export VK_INSTANCE_LAYERS=VK_LAYER_SIMPLE_example
simple-layer-install: simple-layer
	sudo cp src/simple-layer.json /usr/share/vulkan/explicit_layer.d/
	@echo "Now set the following environment variables:"
	@echo "export VK_INSTANCE_LAYERS=VK_LAYER_SIMPLE_example"
	@echo "export LD_LIBRARY_PATH=$(PWD):\$$LD_LIBRARY_PATH"

.PHONY: ctags
ctags: 
	ctags -R  ~/work/ai/vulkan/1.3.283.0/x86_64/include/ .

clean:
	@rm -f simple simple-shader shader list-dev simple-layer.so
