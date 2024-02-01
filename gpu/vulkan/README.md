## Vulkan API Example


### Installation
Install the Vulkan SDK.
```console
$ sudo dnf install @development-tools glm-devel cmake libpng-devel wayland-devel \
libpciaccess-devel libX11-devel libXpresent libxcb xcb-util libxcb-devel libXrandr-devel \
xcb-util-keysyms-devel xcb-util-wm-devel python3 git lz4-devel libzstd-devel python3-distutils-extra qt \
gcc-g++ wayland-protocols-devel ninja-build python3-jsonschema qt5-qtbase-devel

$ sudo dnf install vulkan-tools
$ sudo dnf install glfw-devel
$ sudo dnf install libXi-devel libXxf86vm-devel
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
Verify the install:
```console
$ vkvia
```
