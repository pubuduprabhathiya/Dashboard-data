#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

__global__ void matrix_mul(float *a, float *b, float *c, unsigned width_a,
                           unsigned width_b, unsigned height_a) {

  // Get our global thread ID
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width_b && row < height_a) {
    double dot = 0;

    for (int i = 0; i < width_a; i++)
      dot += a[width_a * row + i] * b[width_b * i + col];

    c[width_b * row + col] = dot;
  }
}

int main(int argc, char *argv[]) {
  // Size of vectors
  unsigned height_a = (argc > 1 ? atoi(argv[1]) : 500);
  unsigned width_a = (argc > 2 ? atoi(argv[2]) : 110);
  unsigned width_b = (argc > 3 ? atoi(argv[3]) : 220);
  unsigned trials = (argc > 2 ? atoi(argv[2]) : 1000);
  unsigned N = height_a * width_b;

  // Host input,outut vectors
  float *h_a, *h_b, *h_c;

  // Device input,output vectors
  float *d_a, *d_b, *d_c;

  // Allocate host memory
  h_a = (float *)malloc(sizeof(float) * width_a * height_a);
  h_b = (float *)malloc(sizeof(float) * width_a * width_b);
  h_c = (float *)malloc(height_a * width_b * sizeof(float));
  // Allocate memory for each vector on GPU
  cudaMalloc(&d_a, sizeof(float) * width_a * height_a);
  cudaMalloc(&d_b, sizeof(float) * width_a * width_b);
  cudaMalloc(&d_c, sizeof(float) * width_b * height_a);

  // Initialize vectors on host
  for (unsigned i = 0; i < height_a * width_a; i++)
    h_a[i] = rand() % 256;

  for (unsigned i = 0; i < width_a * width_b; i++)
    h_b[i] = rand() % 256;

  // Copy host vectors to device
  cudaMemcpy(d_a, h_a, sizeof(float) * width_a * height_a,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(float) * width_a * width_b,
             cudaMemcpyHostToDevice);

  // Number of thread blocks in grid
  unsigned grid_rows = (height_a + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned grid_cols = (width_b + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dim_grid(grid_cols, grid_rows);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);

  // Do a warm up run
  for (unsigned t = 0; t < 100; t++)
    matrix_mul<<<dim_grid, dim_block>>>(d_a, d_b, d_c, width_a, width_b,
                                        height_a);

  cudaDeviceSynchronize();

  // Time the vector addition
  clock_t t = clock();

  // Execute the kernel
  for (unsigned t = 0; t < trials; t++)
    matrix_mul<<<dim_grid, dim_block>>>(d_a, d_b, d_c, width_a, width_b,
                                        height_a);

  // block CPU execution until all previously issued commands on the device have
  // completed
  cudaDeviceSynchronize();

  t = clock() - t;
  // Copy array back to host
  cudaMemcpy(h_c, d_c, sizeof(float) * height_a * width_b,
             cudaMemcpyDeviceToHost);

  for (unsigned i = 0; i < height_a; i++) {
    for (unsigned j = 0; j < width_b; j++) {
      double dot = 0;
      for (unsigned k = 0; k < width_a; k++)
        dot += h_a[i * width_a + k] * h_b[k * width_b + j];
      assert(fabs(h_c[i * width_b + j] - dot) < 1e-8);
    }
  }

  printf("N: %u blockSize: (%d,%d) gridSize: (%d,%d) time %e\n", N, BLOCK_SIZE,
         BLOCK_SIZE, grid_cols, grid_rows,
         (double)t / (CLOCKS_PER_SEC * trials));

  // Release device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Release host memory
  free(h_a), free(h_b), free(h_c);
  return 0;
}
