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
    // List ICD extensions by first getting the count.
    //
    // Notice that is using `InstanceExtension` (as opposed to InstanceLayer which
    // we will see later):
    // vkEnumerateInstanceExtensionProperties
    //            [      ]
    //                ↑
    uint32_t icd_ext_count;
    if (vkEnumerateInstanceExtensionProperties(NULL, &icd_ext_count, NULL) != VK_SUCCESS) {
        printf("Failed to list the ICD extensions\n");
        return -1;
    }
    printf("Found %d instance extensions:\n", icd_ext_count);

    VkExtensionProperties* icd_extensions = malloc(sizeof(VkExtensionProperties) * icd_ext_count);
    if(vkEnumerateInstanceExtensionProperties(NULL, &icd_ext_count, icd_extensions) != VK_SUCCESS) {
        printf("Failed to list the ICD extensions\n");
        return -1;
    }
    for (uint32_t i = 0; i < icd_ext_count; i++) {
        printf("ICD Extension %d: %s\n", i, icd_extensions[i].extensionName);
    }

    // List Layer extensions
    uint32_t layer_count;
    if(vkEnumerateInstanceLayerProperties(&layer_count, NULL) != VK_SUCCESS) {
        printf("Failed to get the number of layers\n");
        return -1;
    }
    printf("Found %d layers:\n", layer_count);

    // Notice that this is a different from the VkExtensionProperties struct
    // which was something I did not notice initially.
    VkLayerProperties* layers = malloc(sizeof(VkLayerProperties) * layer_count);

    if(vkEnumerateInstanceLayerProperties(&layer_count, layers) != VK_SUCCESS) {
        printf("Failed to get the layer properties\n");
        return -1;
    }

    // Now iterate through layers
    for (uint32_t i = 0; i < layer_count; i++) {

        uint32_t layer_ext_count;
        if (vkEnumerateInstanceExtensionProperties(layers[i].layerName, &layer_ext_count, NULL) != VK_SUCCESS) {
            printf("Failed to get the number of extensions for layer %s\n", layers[i].layerName);
            return -1;
        }
        printf("  '%s' provides %d extensions:\n", layers[i].layerName, layer_ext_count);

        if (layer_ext_count == 0) {
            continue;
        }

        // But now also notice that this is the same struct used for the device
        // extensions properties (is easy to mix up when new to Vulkan)
        VkExtensionProperties* layer_extensions = malloc(sizeof(VkExtensionProperties) * layer_ext_count);
        if(vkEnumerateInstanceExtensionProperties(layers[i].layerName, &layer_ext_count, layer_extensions) != VK_SUCCESS) {
            printf("Failed to get the extensions for layer %s\n", layers[i].layerName);
            return -1;
        }

        for (uint32_t j = 0; j < layer_ext_count; j++) {
            printf("    %s\n", layer_extensions[j].extensionName);
        }
        free(layer_extensions);
        printf("\n");
    }

    free(layers);

    VkInstance instance = createInstance();

    uint32_t deviceCount = 0;
    if (vkEnumeratePhysicalDevices(instance, &deviceCount, NULL) != VK_SUCCESS) {
        printf("Failed to enumerate physical devices.\n");
        return -1;
    }

    if (deviceCount == 0) {
        printf("Failed to find GPUs with Vulkan support.\n");
        return -1;
    }

    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
    if (vkEnumeratePhysicalDevices(instance, &deviceCount, devices) != VK_SUCCESS) {
        printf("Failed to enumerate physical devices.\n");
        free(devices);
        return -1;
    }

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

