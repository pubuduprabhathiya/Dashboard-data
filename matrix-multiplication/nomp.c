#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiply(float *a, float *b, float *c, const int size) {
#pragma nomp for transform("transforms", "matrix_transform")
  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      double dot = 0;
      for (unsigned k = 0; k < size; k++)
        dot += a[i * size + k] * b[k * size + j];
      c[i * size + j] = dot;
    }
  }
}

int main(int argc, char **argv) {
  unsigned size = (argc > 1 ? atoi(argv[1]) : 1000);
  unsigned trials = (argc > 2 ? atoi(argv[2]) : 1000);

  srand(time(NULL));
  const int mem_size = size * size;
  float *a = (float *)malloc(size * size * sizeof(float));
  float *b = (float *)malloc(size * size * sizeof(float));
  float *c = (float *)malloc(size * size * sizeof(float));

  for (unsigned i = 0; i < size * size; i++)
    a[i] = (rand() % 256 + 1.0) / RAND_MAX;

  for (unsigned i = 0; i < size * size; i++)
    b[i] = (rand() % 256 + 1.0) / RAND_MAX;

#pragma nomp init(argc, argv)
#pragma nomp update(to : a[0, mem_size], b[0, mem_size])
#pragma nomp update(alloc : c[0, mem_size])

  // Do a warmup run
  for (unsigned i = 0; i < 100; i++)
    multiply(a, b, c, size);
#pragma nomp sync

  // Run the pi integration
  clock_t t = clock();
  for (unsigned t = 0; t < trials; t++)
    multiply(a, b, c, size);
#pragma nomp sync
  t = clock() - t;

#pragma nomp update(from : c[0, mem_size])
#pragma nomp update(free : a[0, mem_size], b[0, mem_size], c[0, mem_size])

  printf("N: %u time: %e\n", size, (double)t / (CLOCKS_PER_SEC * trials));

#pragma nomp finalize

  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      double dot = 0;
      for (unsigned k = 0; k < size; k++)
        dot += a[i * size + k] * b[k * size + j];
      assert(fabs(c[i * size + j] - dot) < 1e-8);
    }
  }

  free(a), free(b), free(c);
  return 0;
}
