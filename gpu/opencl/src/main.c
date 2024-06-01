#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define NUM_ELEMENTS 1024

const char *kernelSource = 
"__kernel void add_vectors(__global const float *a, __global const float *b, __global float *result, unsigned int numElements) {\n"
"    int id = get_global_id(0);\n"
"    if (id < numElements) {\n"
"        result[id] = a[id] + b[id];\n"
"    }\n"
"}\n";

void printPlatformInfo(cl_platform_id platform) {
    char info[1024];
    size_t infoSize;

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(info), info, &infoSize);
    printf("\n-------------------------------\n");
    printf("Platform Name: %s\n", info);

    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(info), info, &infoSize);
    printf("Platform Vendor: %s\n", info);

    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(info), info, &infoSize);
    printf("Platform Version: %s\n", info);

    clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, sizeof(info), info, &infoSize);
    printf("Platform Profile: %s\n", info);
    printf("-------------------------------\n");
}

int main() {
    printf("OpenCL Exploration...!\n");

    cl_int err;

    // Query the number of available platforms
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get the number of platforms: cl_err: %d\n", err);
        return 1;
    }
    printf("Number of platforms: %d\n", numPlatforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get the number array or cl_platform_id: cl_err: %d\n", err);
        return 1;
    }

    for (int i = 0; i < numPlatforms; i++) {
        printPlatformInfo(platforms[i]);
    }

    cl_device_id device;
    err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get device id: cl_err: %d\n", err);
        return 1;
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create context: cl_err: %d\n", err);
        return 1;
    }

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create command queue: cl_err: %d\n", err);
        return 1;
    }

    unsigned int numElements = NUM_ELEMENTS;

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create program: cl_err: %d\n", err);
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print the build log if there's an error
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build error:\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "add_vectors", &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create kernel: cl_err: %d\n", err);
        return 1;
    }

    float *a = (float *)malloc(NUM_ELEMENTS * sizeof(float));
    float *b = (float *)malloc(NUM_ELEMENTS * sizeof(float));
    float *result = (float *)malloc(NUM_ELEMENTS * sizeof(float));

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     NUM_ELEMENTS * sizeof(float), a, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create bufferA: cl_err: %d\n", err);
        return 1;
    }
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     NUM_ELEMENTS * sizeof(float), b, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create bufferB: cl_err: %d\n", err);
        return 1;
    }
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          NUM_ELEMENTS * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create bufferResult: cl_err: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &numElements);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments: cl_err: %d\n", err);
        return 1;
    }

    size_t globalSize = NUM_ELEMENTS;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to enqueue kernel: cl_err: %d\n", err);
        return 1;
    }


    err = clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0,
                        NUM_ELEMENTS * sizeof(float), result, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read buffer: cl_err: %d\n", err);
        return 1;
    }


    for (int i = 0; i < 10; i++) {
        printf("%.2f + %.2f = %.2f\n", a[i], b[i], result[i]);
    }

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(a);
    free(b);
    free(result);

    return 0;
}
