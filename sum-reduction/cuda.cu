#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define SHMEM_SIZE 256 * sizeof(double)
#define SIZE 256

__device__ void warp_reduce(volatile double *sdata, unsigned int tid) {
  if (blockDim.x >= 64)
    sdata[tid] += sdata[tid + 32];
  if (blockDim.x >= 32)
    sdata[tid] += sdata[tid + 16];
  if (blockDim.x >= 16)
    sdata[tid] += sdata[tid + 8];
  if (blockDim.x >= 8)
    sdata[tid] += sdata[tid + 4];
  if (blockDim.x >= 4)
    sdata[tid] += sdata[tid + 2];
  if (blockDim.x >= 2)
    sdata[tid] += sdata[tid + 1];
}

__global__ void sum_reduction(double *v, double *v_r, unsigned int n) {

  // Allocate shared memory
  __shared__ double partial_sum[SHMEM_SIZE];

  unsigned int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    partial_sum[tid] = v[i];
  else
    partial_sum[tid] = 0;
  __syncthreads();

  if (blockDim.x >= 512) {
    if (tid < 256) {
      partial_sum[tid] += partial_sum[tid + 256];
    }
    __syncthreads();
  }
  if (blockDim.x >= 256) {
    if (tid < 128) {
      partial_sum[tid] += partial_sum[tid + 128];
    }
    __syncthreads();
  }
  if (blockDim.x >= 128) {
    if (tid < 64) {
      partial_sum[tid] += partial_sum[tid + 64];
    }
    __syncthreads();
  }
  if (tid < 32)
    warp_reduce(partial_sum, tid);

  if (tid == 0)
    v_r[blockIdx.x] = partial_sum[0];
}

int main(int argc, char *argv[]) {
  // vector size
  int n = (argc > 1 ? atoi(argv[1]) : 1 << 16);
  int trials = (argc > 2 ? atoi(argv[2]) : 1000);
  size_t bytes = n * sizeof(double);
  double sequential_sum = 0;

  // TB Size
  int TB_SIZE = SIZE;

  // Grid Size
  int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE;

  // Original vector and result vector
  double *h_v, *h_v_r;
  double *d_v, *d_v_r;

  // Allocate memory
  h_v = (double *)malloc(bytes);
  h_v_r = (double *)malloc(sizeof(double) * GRID_SIZE);
  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  // Initialize vector
  for (int i = 0; i < n; i++) {
    h_v[i] = (double)rand();
    sequential_sum += h_v[i];
  }

  // Copy to device
  cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

  // Do a warm up run
  for (unsigned t = 0; t < 100; t++)
    sum_reduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r, n);

  // Time the vector addition
  clock_t t = clock();

  for (unsigned t = 0; t < trials; t++)
    sum_reduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r, n);

  cudaDeviceSynchronize();

  // Copy to host;
  cudaMemcpy(h_v_r, d_v_r, sizeof(double) * GRID_SIZE, cudaMemcpyDeviceToHost);

  double sum = 0;
  for (int i = 0; i < GRID_SIZE; i++)
    sum += h_v_r[i];

  t = clock() - t;

  assert(fabs(sum - sequential_sum) < 1e-12);

  printf("N: %d blockSize: %d gridSize: %d time %e\n", n, TB_SIZE, GRID_SIZE,
         (double)t / (CLOCKS_PER_SEC * trials));

  // Release device memory
  cudaFree(d_v);
  cudaFree(d_v_r);

  // Release host memory
  free(h_v);
  free(h_v_r);

  return 0;
}
