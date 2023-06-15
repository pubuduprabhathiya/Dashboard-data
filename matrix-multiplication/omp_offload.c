#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int height_a = (argc > 1 ? atoi(argv[1]) : 500);
  int width_a = (argc > 2 ? atoi(argv[2]) : 110);
  int width_b = (argc > 3 ? atoi(argv[3]) : 220);
  int trials = (argc > 4 ? atoi(argv[4]) : 1000);

  double *a = (double *)malloc(width_a * height_a * sizeof(double));
  double *b = (double *)malloc(width_a * width_b * sizeof(double));
  double *c = (double *)malloc(width_b * height_a * sizeof(double));

  for (int i = 0; i < height_a * width_a; i++)
    a[i] = fmod((double)rand(), 256);

  for (int i = 0; i < width_a * width_b; i++)
    b[i] = fmod((double)rand(), 256);

#pragma omp target enter data map(to                                           \
                                  : a [0:width_a * height_a],                  \
                                    b [0:width_a * width_b])                   \
    map(alloc                                                                  \
        : c [0:width_b * height_a])

  // warm up runs
  for (int t = 0; t < 100; t++) {
#pragma omp target teams distribute parallel for
    for (int i = 0; i < height_a * width_b; i++) {
      c[i] = 0;
      int r = i / width_b, col = i - r * width_b;
      for (int k = 0; k < width_a; k++)
        c[i] += a[r * width_a + k] * b[k * width_b + col];
    }
  }

  double t = omp_get_wtime();
  for (int t = 0; t < trials; t++) {
#pragma omp target teams distribute parallel for
    for (int i = 0; i < height_a * width_b; i++) {
      c[i] = 0;
      int r = i / width_b, col = i - r * width_b;
      for (int k = 0; k < width_a; k++)
        c[i] += a[r * width_a + k] * b[k * width_b + col];
    }
  }
  t = omp_get_wtime() - t;

#pragma omp target exit data map(from : c [0:width_b * height_a])
#pragma omp target exit data map(delete                                        \
                                 : a [0:width_a * height_a],                   \
                                   b [0:width_a * width_b],                    \
                                   c [0:width_b * height_a])

  printf("N: %d time %e \n", height_a * width_b, t / trials);

  for (int i = 0; i < height_a; i++) {
    for (int j = 0; j < width_b; j++) {
      double dot = 0;
      for (int k = 0; k < width_a; k++)
        dot += a[i * width_a + k] * b[k * width_b + j];
      assert(fabs(c[i * width_b + j] - dot) < 1e-8);
    }
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
