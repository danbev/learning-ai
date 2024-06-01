// main.c
#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define NUM_ELEMENTS 1024

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


    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, NULL);

    unsigned int numElements = NUM_ELEMENTS;


    const char *source =
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        "#include \"add_vectors.cl\"\n";

    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "add_vectors", NULL);

    float *a = (float *)malloc(NUM_ELEMENTS * sizeof(float));
    float *b = (float *)malloc(NUM_ELEMENTS * sizeof(float));
    float *result = (float *)malloc(NUM_ELEMENTS * sizeof(float));

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     NUM_ELEMENTS * sizeof(float), a, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     NUM_ELEMENTS * sizeof(float), b, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          NUM_ELEMENTS * sizeof(float), NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &numElements);

    size_t globalSize = NUM_ELEMENTS;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0,
                        NUM_ELEMENTS * sizeof(float), result, 0, NULL, NULL);

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
