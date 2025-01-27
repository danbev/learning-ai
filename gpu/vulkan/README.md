## Vulkan API Example


### Installation
Install the Vulkan SDK.

#### Fedora
```console
```console
$ sudo dnf install @development-tools glm-devel cmake libpng-devel wayland-devel \
libpciaccess-devel libX11-devel libXpresent libxcb xcb-util libxcb-devel libXrandr-devel \
xcb-util-keysyms-devel xcb-util-wm-devel python3 git lz4-devel libzstd-devel python3-distutils-extra qt \
gcc-g++ wayland-protocols-devel ninja-build python3-jsonschema qt5-qtbase-devel

$ sudo dnf install vulkan-tools
$ sudo dnf install glfw-devel
$ sudo dnf install libXi-devel libXxf86vm-devel
```

#### Ubuntu 22.04:
```console
$ wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
$ sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
$ sudo apt update
$ sudo apt install vulkan-sdk
```

```console
$ dnf list installed | grep vulkan
mesa-vulkan-drivers.x86_64                             23.3.1-4.fc39                       @updates              
vulkan-headers.noarch                                  1.3.268.0-1.fc39                    @updates              
vulkan-loader.x86_64                                   1.3.268.0-1.fc39                    @updates              
vulkan-loader-devel.x86_64                             1.3.268.0-1.fc39                    @updates
```

Download the SDK from: 
https://vulkan.lunarg.com/sdk/home#linux
Unpack the SDK and use the version that you downloaded:.
```console
$ tar xvf vulkansdk-linux-x86_64-1.3.275.0.tar.xz
```

```console
$ source /home/danielbevenius/work/ai/vulkan/1.3.275.0/setup-env.sh
```
```console
$ sudo apt install libglm-dev
```

Verify the install:
```console
$ vkvia
```

### Ubuntu 24.04
```console
$ wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
$ sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list http://packages.lunarg.com/vulkan/lunarg-vulkan-unstable.list


$ sudo apt install vulkan-tools libvulkan1 libvulkan-dev
