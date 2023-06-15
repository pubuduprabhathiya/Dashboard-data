#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  unsigned int N = (argc > 1 ? atoi(argv[1]) : 1 << 16);
  unsigned trials = (argc > 2 ? atoi(argv[2]) : 1000);

  double *a = (double *)malloc(N * sizeof(double));
  double seq_sum = 0;
  double sum;

  for (int i = 0; i < N; i++) {
    a[i] = (double)rand();
    seq_sum += a[i];
  }

#pragma omp target enter data map(to : a [0:N])

  // warm up runs
  for (int t = 0; t < 100; t++) {
    sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++)
      sum += a[i];
  }

  double t = omp_get_wtime();
  for (int j = 0; j < trials; j++) {
    sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++)
      sum += a[i];
  }
  t = omp_get_wtime() - t;
#pragma omp target exit data map(delete : a [0:N])

  printf("N: %d time %e \n", N, t / trials);

  assert(fabs(sum - seq_sum) < 1e-12);

  free(a);
  return 0;
}
