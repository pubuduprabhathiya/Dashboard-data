#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SOURCE_SIZE 0x100000

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char *knl_source =
    "__kernel void integrate(__global double *output, const double step,  \n"
    "                        const int steps, __local double *cache) {    \n"
    "  const int global_id = get_global_id(0);                            \n"
    "  const int local_id = get_local_id(0);                              \n"
    "  const int local_size = get_local_size(0);                          \n"
    "                                                                     \n"
    "  // Compute the single value of the integration                     \n"
    "  if (global_id < steps) {                                           \n"
    "    double x = (global_id + 0.5) * step;                             \n"
    "    cache[local_id] = 4.0 / (1.0 + x * x);                           \n"
    "  } else {                                                           \n"
    "    cache[local_id] = 0;                                             \n"
    "  }                                                                  \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                      \n"
    "                                                                     \n"
    "  // Calculate the sum of the local work group                       \n"
    "  for (int offset = local_size / 2; offset > 0; offset /= 2) {       \n"
    "    if (local_id < offset) {                                         \n"
    "      cache[local_id] += cache[local_id + offset];                   \n"
    "    }                                                                \n"
    "    barrier(CLK_LOCAL_MEM_FENCE);                                    \n"
    "  }                                                                  \n"
    "                                                                     \n"
    "  // Update the cumulative sum of the local group in the global cache\n"
    "  if (local_id == 0) {                                               \n"
    "    output[global_id / local_size] = cache[0];                       \n"
    "  }                                                                  \n"
    "}                                                                    \n";

int main(int argc, char *argv[]) {
  unsigned int steps = (argc > 1 ? atoi(argv[1]) : 1000000);
  unsigned int trials = (argc > 2 ? atoi(argv[2]) : 1000);

  cl_platform_id platform; // OpenCL platform
  cl_device_id device_id;  // device ID
  cl_context context;      // context
  cl_command_queue queue;  // command queue
  cl_program program;      // program
  cl_kernel kernel;        // kernel

  size_t global_size, local_size;
  cl_int err;

  // Number of work items in each local work group
  local_size = 64;

  // Number of total work items - local_size must be devisor
  global_size = ((steps + local_size - 1) / local_size) * local_size;
  double step = 1.0 / (double)global_size;

  // Bind to platform
  err = clGetPlatformIDs(1, &platform, NULL);

  // Get ID for the device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);

  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  // Create a command queue
  queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **)&knl_source,
                                      NULL, &err);

  // Build the program executable
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  // Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "integrate", &err);

  // Create the output buffer to store the sum of each workgroup
  unsigned int output_memory_size = (steps + local_size - 1) / local_size;
  size_t output_memory_bytes = sizeof(double) * output_memory_size;
  cl_mem output_memory = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        output_memory_bytes, NULL, &err);
  double *results = (double *)malloc(output_memory_bytes);

  // Set the arguments to our compute kernel
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&output_memory);
  err |= clSetKernelArg(kernel, 1, sizeof(double), (void *)&step);
  err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), (void *)&steps);
  err |= clSetKernelArg(kernel, 3, sizeof(double) * local_size, NULL);

  // Do a warm up run
  for (unsigned t = 0; t < 100; t++) {
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                 &local_size, 0, NULL, NULL);
    clFinish(queue);
  }

  // Execute the kernel
  double pi;
  clock_t t = clock();
  for (unsigned j = 0; j < trials; j++) {
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                 &local_size, 0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, output_memory, CL_TRUE, 0, output_memory_bytes,
                        results, 0, NULL, NULL);

    // Calculate the pi value
    double sum = 0;
    for (int i = 0; i < output_memory_size; i++)
      sum += results[i];

    pi = step * sum;
  }

  t = clock() - t;
  printf("N: %u local: %lu global: %lu pi: %.3f time: %e\n", steps, local_size,
         global_size, pi, (double)t / (CLOCKS_PER_SEC * trials));

  // Release OpenCL resources
  clReleaseMemObject(output_memory);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // Release host memory
  free(results);

  return 0;
}
