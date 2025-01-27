## WebGPU exploration


### Configuration/setup
First enable WebGPU in `chrome://flags` and search for `WebGPU` and enable the
flags. We can then inspect status using `chrome://gpu`.

Then start chrome with the following flag:
```
$ google-chrome --enable-features=Vulkan
```
I added this to `/usr/share/applications/google-chrome.desktop`.
And after restarting chrome, we can see the following in `chrome://gpu`:
```
*   Vulkan: Enabled
```

There is a html file with a javascript check block that can be used to check if
WebGPU is enabled in the browser:
```console
$ open src/check.html
```

### Run the example
```console
$ open src/matrix-mul.html
```
