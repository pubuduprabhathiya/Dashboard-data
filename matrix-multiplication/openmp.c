#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

  unsigned height_a = (argc > 1 ? atoi(argv[1]) : 500);
  unsigned width_a = (argc > 2 ? atoi(argv[2]) : 110);
  unsigned width_b = (argc > 3 ? atoi(argv[3]) : 220);
  unsigned trials = (argc > 4 ? atoi(argv[4]) : 1000);

  float *a, *b, *c;

  a = (float *)malloc(width_a * height_a * sizeof(float));
  b = (float *)malloc(width_a * width_b * sizeof(float));
  c = (float *)malloc(width_b * height_a * sizeof(float));

  for (unsigned i = 0; i < height_a * width_a; i++)
    a[i] = rand() % 256;

  for (unsigned i = 0; i < width_a * width_b; i++)
    b[i] = rand() % 256;

  unsigned N = height_a * width_b;

#pragma omp parallel
  {
    unsigned thrds = omp_get_num_threads();
    unsigned tid = omp_get_thread_num();
    unsigned size = N / thrds;
    unsigned rem = N - size * thrds;
    size += (tid < rem);
    unsigned start = (tid < rem ? tid * size : tid * size + rem);
    unsigned end = start + size;

    // Do a warm up run
    for (unsigned t = 0; t < 100; t++) {
      for (unsigned i = start; i < end; i++) {
        unsigned dot = 0;
        unsigned r = i / width_b, col = i - r * width_b;
        for (unsigned k = 0; k < width_a; k++)
          dot += a[r * width_a + k] * b[k * width_b + col];
        c[i] = dot;
      }
    }

    // Now time the matrix multiplication operation
    double t = omp_get_wtime();

    for (unsigned t = 0; t < trials; t++) {
      for (unsigned i = start; i < end; i++) {
        unsigned dot = 0;
        unsigned r = i / width_b, col = i - r * width_b;
        for (unsigned k = 0; k < width_a; k++)
          dot += a[r * width_a + k] * b[k * width_b + col];
        c[i] = dot;
      }
    }

    t = omp_get_wtime() - t;

    if (tid == 0)
      printf("N: %u # threads: %u time: %e\n", N, thrds, t / trials);
  }

  for (unsigned i = 0; i < height_a; i++) {
    for (unsigned j = 0; j < width_b; j++) {
      double dot = 0;
      for (unsigned k = 0; k < width_a; k++)
        dot += a[i * width_a + k] * b[k * width_b + j];
      assert(fabs(c[i * width_b + j] - dot) < 1e-8);
    }
  }

  free(a), free(b), free(c);
  return 0;
}
