#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <stdio.h>
#include <string.h>

static PFN_vkGetInstanceProcAddr g_next_gipa = NULL;
static PFN_vkCreateInstance      g_next_CreateInstance = NULL;
static PFN_vkDestroyInstance     g_next_DestroyInstance = NULL;

static const VkLayerInstanceCreateInfo* find_link_info(const VkInstanceCreateInfo* ci) {
    const VkLayerInstanceCreateInfo* chain = (const VkLayerInstanceCreateInfo*)ci->pNext;

    while (chain) {
        if (chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
            chain->function == VK_LAYER_LINK_INFO) {
            return chain;
        }
        // Extract next layer's function pointer from pNext chain
        chain = (const VkLayerInstanceCreateInfo*) chain->pNext;
    }

    return NULL;
}

/*
 * This is the intercept function for vkCreateInstance.
 */
VKAPI_ATTR VkResult VKAPI_CALL simple_vkCreateInstance(const VkInstanceCreateInfo* create_info,
    const VkAllocationCallbacks* allocator,
    VkInstance* instance) {
    printf("[SIMPLE_LAYER] Creating Vulkan instance!\n");

    const VkLayerInstanceCreateInfo* link_info = find_link_info(create_info);
    if (!link_info) {
        fprintf(stderr, "[SIMPLE_LAYER] ERROR: Missing VK_LAYER_LINK_INFO\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // gipa stands for GetInstanceProcAddr
    g_next_gipa = link_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;

    g_next_CreateInstance = (PFN_vkCreateInstance) g_next_gipa(NULL, "vkCreateInstance");

    if (!g_next_CreateInstance) {
        fprintf(stderr, "[SIMPLE_LAYER] ERROR: Failed to fetch next vkCreateInstance\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Call the real vkCreateInstance (or the next layer's)
    VkResult result = g_next_CreateInstance(create_info, allocator, instance);

    if (result == VK_SUCCESS) {
        printf("[SIMPLE_LAYER] Instance created successfully!\n");
        // After we have a real instance, fetch next vkDestroyInstance.
        g_next_DestroyInstance = (PFN_vkDestroyInstance) g_next_gipa(*instance, "vkDestroyInstance");
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL simple_vkDestroyInstance( VkInstance instance, const VkAllocationCallbacks* allocator) {
    printf("[SIMPLE_LAYER] Destroying Vulkan instance!\n");
    if (g_next_DestroyInstance) {
        g_next_DestroyInstance(instance, allocator);
    } else if (g_next_gipa) {
        PFN_vkDestroyInstance next = (PFN_vkDestroyInstance)g_next_gipa(instance, "vkDestroyInstance");
        if (next) {
            next(instance, allocator);
        }
    }
}

/*
 * When an application calls vkCreateInstance, the loader will call
 * vkGetInstanceProcAddr(NULL, "vkCreateInstance") and this allows a layer
 * implementation to intercept that call and return its own function pointer.
 */
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL simple_vkGetInstanceProcAddr(VkInstance instance, const char* name) {
    if (strcmp(name, "vkCreateInstance") == 0)
        return (PFN_vkVoidFunction)simple_vkCreateInstance;
    if (strcmp(name, "vkDestroyInstance") == 0)
        return (PFN_vkVoidFunction)simple_vkDestroyInstance;
    if (strcmp(name, "vkGetInstanceProcAddr") == 0)
        return (PFN_vkVoidFunction)simple_vkGetInstanceProcAddr;

    return g_next_gipa ? g_next_gipa(instance, name) : NULL;
}

/*
 * When the Vulkan loader reads simple_layer.json, it will load the shared object
 * into memory and after that call this function. This will provide the loader
 * with the version that this layer supports and also notice that it will get
 * the entrypoint, which is the vkGetInstanceProcAddr function for this layer.
 */ 
VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* negotiation) {
    printf("[SIMPLE_LAYER] Negotiating layer interface\n");
    if (!negotiation || negotiation->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;

    // We support interface version 2
    if (negotiation->loaderLayerInterfaceVersion < 2)
        return VK_ERROR_INITIALIZATION_FAILED;

    negotiation->loaderLayerInterfaceVersion = 2;
    negotiation->pfnGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) simple_vkGetInstanceProcAddr;
    negotiation->pfnGetDeviceProcAddr = NULL;
    negotiation->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}
