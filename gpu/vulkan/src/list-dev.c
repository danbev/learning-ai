#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>

// Initialize Vulkan instance
VkInstance createInstance() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "DeviceLister";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
        printf("Failed to create Vulkan instance.\n");
        exit(-1);
    }

    return instance;
}

int main() {
    VkInstance instance = createInstance();

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);

    if (deviceCount == 0) {
        printf("Failed to find GPUs with Vulkan support.\n");
        return -1;
    }

    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

    printf("Found %d device(s)\n\n", deviceCount);

    for (uint32_t i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);

        printf("Device %d: %s\n", i, deviceProperties.deviceName);
        printf("Device ID: %d\n", deviceProperties.deviceID);
        printf("Vendor ID: %d\n", deviceProperties.vendorID);
        printf("\n");
    }

    free(devices);
    vkDestroyInstance(instance, NULL);

    return 0;
}

