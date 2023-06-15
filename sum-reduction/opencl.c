#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
#include <CL/cl.h>
#endif

// OpenCL kernel
const char *knl_source =
    "__kernel void sum_reduction(__global double *v,                        \n"
    "                     __global double *v_r,                             \n"
    "                     const unsigned n,                                 \n"
    "                     __local volatile double *smemory)                 \n"
    "{                                                                      \n"
    "                                                                       \n"
    "    unsigned int tid = get_local_id(0);                                \n"
    "    unsigned int i =                                                     "
    "                get_group_id(0) * get_local_size(0) + get_local_id(0); \n"
    "                                                                       \n"
    "    if (i < n)                                                         \n"
    "       smemory[tid] = v[i];                                            \n"
    "    else                                                               \n"
    "       smemory[tid] = 0;                                               \n"
    "    barrier(CLK_LOCAL_MEM_FENCE);                                      \n"
    "                                                                       \n"
    "    if (get_local_size(0) >= 512){                                     \n"
    "       if (tid < 256)                                                  \n"
    "           smemory[tid] += smemory[tid + 256];                         \n"
    "       barrier(CLK_LOCAL_MEM_FENCE);                                   \n"
    "    }                                                                  \n"
    "    if (get_local_size(0) >= 256){                                     \n"
    "       if (tid < 128)                                                  \n"
    "           smemory[tid] += smemory[tid + 128];                         \n"
    "       barrier(CLK_LOCAL_MEM_FENCE);                                   \n"
    "    }                                                                  \n"
    "    if (get_local_size(0) >= 128){                                     \n"
    "       if (tid < 64)                                                   \n"
    "           smemory[tid] += smemory[tid + 64];                          \n"
    "       barrier(CLK_LOCAL_MEM_FENCE);                                   \n"
    "    }                                                                  \n"
    "                                                                       \n"
    "    if (tid < 32){                                                     \n"
    "       if (get_local_size(0) >= 64)                                    \n"
    "           smemory[tid] += smemory[tid + 32];                          \n"
    "       if (get_local_size(0) >= 32)                                    \n"
    "           smemory[tid] += smemory[tid + 16];                          \n"
    "       if (get_local_size(0) >= 16)                                    \n"
    "           smemory[tid] += smemory[tid + 8];                           \n"
    "       if (get_local_size(0) >= 8)                                     \n"
    "           smemory[tid] += smemory[tid + 4];                           \n"
    "       if (get_local_size(0) >= 4)                                     \n"
    "           smemory[tid] += smemory[tid + 2];                           \n"
    "       if (get_local_size(0) >= 2)                                     \n"
    "           smemory[tid] += smemory[tid + 1];                           \n"
    "    }                                                                  \n"
    "    if (tid == 0)                                                      \n"
    "        v_r[get_group_id(0)] = smemory[0];                             \n"
    "}                                                                      \n"
    "                                                                       \n";

int main(int argc, char *argv[]) {
  unsigned N = (argc > 1 ? atoi(argv[1]) : 1 << 16);
  unsigned trials = (argc > 2 ? atoi(argv[2]) : 1000);

  size_t global_size, local_size;
  cl_int err;

  // Number of work items in each local work group
  local_size = 64;

  // Number of total work items - local_size must be divisor
  unsigned num_blocks = (N + local_size - 1) / local_size;
  global_size = num_blocks * local_size;

  double *v = (double *)malloc(N * sizeof(double));
  double *v_r = (double *)malloc(N * sizeof(double));

  // Device input buffers
  cl_mem d_v, d_v_r;

  cl_platform_id platform; // OpenCL platform
  cl_device_id device_id;  // device ID
  cl_context context;      // context
  cl_command_queue queue;  // command queue
  cl_program program;      // program
  cl_kernel kernel;        // kernel

  // Initialize vectors on host
  double seq_sum = 0;
  double sum = 0;
  for (unsigned i = 0; i < N; i++) {
    v[i] = (double)rand();
    seq_sum += v[i];
  }

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
  kernel = clCreateKernel(program, "sum_reduction", &err);

  // Create the input and output arrays in device memory for our calculation
  // Size, in bytes, of each vector
  size_t i_bytes = N * sizeof(double);
  size_t r_bytes = num_blocks * sizeof(double);
  d_v = clCreateBuffer(context, CL_MEM_READ_ONLY, i_bytes, NULL, NULL);
  d_v_r = clCreateBuffer(context, CL_MEM_WRITE_ONLY, r_bytes, NULL, NULL);

  // Write our data set into the input array in device memory
  err = clEnqueueWriteBuffer(queue, d_v, CL_TRUE, 0, i_bytes, v, 0, NULL, NULL);

  // Set the arguments to our compute kernel
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_v);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_v_r);
  err |= clSetKernelArg(kernel, 2, sizeof(unsigned), &N);
  err |= clSetKernelArg(kernel, 3, sizeof(double) * local_size, NULL);

  // Do a warm up run
  for (unsigned t = 0; t < 100; t++)
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                 &local_size, 0, NULL, NULL);
  clFinish(queue);

  // Time the vector addition
  clock_t t = clock();
  for (unsigned t = 0; t < trials; t++)
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                 &local_size, 0, NULL, NULL);

  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);

  // Read the results from the device
  clEnqueueReadBuffer(queue, d_v_r, CL_TRUE, 0, r_bytes, v_r, 0, NULL, NULL);

  for (unsigned i = 0; i < num_blocks; i++)
    sum += v_r[i];
  t = clock() - t;

  assert(fabs(sum - seq_sum) < 1e-12);

  printf("N: %u local: %lu global: %lu time: %e\n", N, local_size, global_size,
         (double)t / (CLOCKS_PER_SEC * trials));

  // release OpenCL resources
  clReleaseMemObject(d_v);
  clReleaseMemObject(d_v_r);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory
  free(v), free(v_r);
  return 0;
}
