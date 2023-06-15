#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 64

__global__ void integrate(double *output, const double step, const int steps) {
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_id = threadIdx.x;
  __shared__ double cache[BLOCK_SIZE];

  // Compute the single value of the integration
  if (global_id < steps) {
    double x = (global_id + 0.5) * step;
    cache[local_id] = 4.0 / (1.0 + x * x);
  } else {
    cache[local_id] = 0;
  }
  __syncthreads();

  // Calculate the sum of the local work group
  for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
    if (local_id < offset) {
      cache[local_id] += cache[local_id + offset];
    }
    __syncthreads();
  }

  // Update the cumulative sum of the local group in the global cache
  if (local_id == 0) {
    output[global_id / BLOCK_SIZE] = cache[0];
  }
}

int main(int argc, char *argv[]) {
  unsigned steps = (argc > 1 ? atoi(argv[1]) : 1000000);
  unsigned trials = (argc > 2 ? atoi(argv[2]) : 1000);

  // Initialize the block and grid dimensions
  size_t grid_size, global_size;
  double *results, *output_memory;

  grid_size = (steps + BLOCK_SIZE - 1) / BLOCK_SIZE;
  global_size = BLOCK_SIZE * grid_size;
  double step = 1.0 / (double)(global_size);

  // Initialize the memory buffers on both host and device
  size_t output_memory_bytes = grid_size * sizeof(double);
  results = (double *)malloc(output_memory_bytes);
  cudaMalloc((void **)&output_memory, output_memory_bytes);

  // Do a warm up run
  for (unsigned t = 0; t < 100; t++) {
    integrate<<<grid_size, BLOCK_SIZE>>>(output_memory, step, steps);
    cudaDeviceSynchronize();
  }

  // Execute the kernel
  double pi;
  clock_t t = clock();
  for (unsigned t = 0; t < trials; t++) {
    integrate<<<grid_size, BLOCK_SIZE>>>(output_memory, step, steps);
    cudaDeviceSynchronize();

    // Copy the value from the device to host
    cudaMemcpy(results, output_memory, output_memory_bytes,
               cudaMemcpyDeviceToHost);

    // Calculate the pi value
    double sum = 0;
    for (int i = 0; i < grid_size; i++)
      sum += results[i];
    pi = step * sum;
  }

  t = clock() - t;
  printf("N: %u local: %i grid: %lu pi: %.3f time: %e\n", steps, BLOCK_SIZE,
         grid_size, pi, (double)t / (CLOCKS_PER_SEC * trials));

  // Release the resources
  free(results);
  cudaFree(output_memory);

  return 0;
}
