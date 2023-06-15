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
#define LOCAL_SIZE 16

// OpenCL kernel. Each work item takes care of one element of c
const char *knl_source =
    "__kernel void matrix_mul(const unsigned  width_a,               \n"
    "                        const unsigned  height_a,               \n"
    "                        const unsigned  width_b,                \n"
    "                        __global float *a,                      \n"
    "                        __global float *b,                      \n"
    "                        __global float *c)                      \n"
    "{                                                               \n"
    "    //Get our global thread ID                                  \n"
    "    int col = get_global_id(0);                                 \n"
    "    int row = get_global_id(1);                                 \n"
    "                                                                \n"
    "    double dot=0;                                               \n"
    "    if( col < width_b && row < height_a)                        \n"
    "    {                                                           \n"
    "     for(int i=0;i<width_a;i++)                                 \n"
    "       dot+=a[width_a*row+i]*b[width_b*i+col];                  \n"
    "     c[width_b*row+col]=dot;                                    \n"
    "    }                                                           \n"
    "}                                                               \n";

int main(int argc, char *argv[]) {

  unsigned height_a = (argc > 1 ? atoi(argv[1]) : 500);
  unsigned width_a = (argc > 2 ? atoi(argv[2]) : 110);
  unsigned width_b = (argc > 3 ? atoi(argv[3]) : 220);
  unsigned trials = (argc > 4 ? atoi(argv[4]) : 1000);

  // Allocate memory for each matrix on host
  float *h_a, *h_b, *h_c;

  h_a = (float *)malloc(height_a * width_a * sizeof(float));
  h_b = (float *)malloc(width_a * width_b * sizeof(float));
  h_c = (float *)calloc(height_a * width_b, sizeof(float));

  // Initialize matrix on host
  for (unsigned i = 0; i < height_a * width_a; i++)
    h_a[i] = rand() % 256;

  for (unsigned i = 0; i < width_a * width_b; i++)
    h_b[i] = rand() % 256;

  unsigned N = height_a * width_b;

  // Device input buffers
  cl_mem d_a, d_b, d_c;

  cl_platform_id platform; // OpenCL platform
  cl_device_id device_id;  // device ID
  cl_context context;      // context
  cl_command_queue queue;  // command queue
  cl_program program;      // program
  cl_kernel kernel;        // kernel

  cl_int err;

  unsigned grid_rows = ((height_a + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
  unsigned grid_cols = ((width_b + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
  size_t local_size[2] = {LOCAL_SIZE, LOCAL_SIZE};
  size_t global_size[2] = {grid_cols, grid_rows};

  // Bind to platform
  err = clGetPlatformIDs(1, &platform, NULL);

  // Get ID for the device
  err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);

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
  kernel = clCreateKernel(program, "matrix_mul", &err);

  // Create the input and output arrays in device memory for our calculation
  // Size, in bytes, of each vector
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
                       height_a * width_a * sizeof(float), NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
                       width_a * width_b * sizeof(float), NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                       height_a * width_b * sizeof(float), NULL, NULL);

  // Write our data set into the input array in device memory
  err |= clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                              height_a * width_a * sizeof(float), h_a, 0, NULL,
                              NULL);
  err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                              width_a * width_b * sizeof(float), h_b, 0, NULL,
                              NULL);

  // Set the arguments to our compute kernel
  err |= clSetKernelArg(kernel, 0, sizeof(unsigned), &width_a);
  err |= clSetKernelArg(kernel, 1, sizeof(unsigned), &height_a);
  err |= clSetKernelArg(kernel, 2, sizeof(unsigned), &width_b);
  err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_c);

  // Do a warm up run
  for (unsigned t = 0; t < 100; t++)
    err |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size,
                                  local_size, 0, NULL, NULL);
  clFinish(queue);

  // Time the matrix multiplication
  clock_t t = clock();
  for (unsigned t = 0; t < trials; t++)
    err |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size,
                                  local_size, 0, NULL, NULL);

  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);
  t = clock() - t;

  // Read the results from the device
  err |= clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                             height_a * width_b * sizeof(float), h_c, 0, NULL,
                             NULL);

  if (err != CL_SUCCESS) {
    printf("OpenCL error executing kernel: %d\n", err);
    goto cleanup;
  }

  for (unsigned i = 0; i < height_a; i++) {
    for (unsigned j = 0; j < width_b; j++) {
      double dot = 0;
      for (unsigned k = 0; k < width_a; k++)
        dot += h_a[i * width_a + k] * h_b[k * width_b + j];
      assert(fabs(h_c[i * width_b + j] - dot) < 1e-8);
    }
  }

  printf("N: %u local:(%lu,%lu) global:(%lu,%lu) time: %e\n", N, local_size[0],
         local_size[1], global_size[0], global_size[1],
         (double)t / (CLOCKS_PER_SEC * trials));
cleanup:
  // release OpenCL resources
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);

  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory
  free(h_a), free(h_b), free(h_c);
  return 0;
}
