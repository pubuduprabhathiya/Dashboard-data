#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  unsigned int N = (argc > 1 ? atoi(argv[1]) : 1 << 16);
  unsigned trials = (argc > 2 ? atoi(argv[2]) : 1000);

  double *a = (double *)malloc(N * sizeof(double));
  double seq_sum = 0, sum;

  for (int i = 0; i < N; i++) {
    a[i] = (double)rand();
    seq_sum += a[i];
  }

#pragma omp parallel
  {
    unsigned threads = omp_get_num_threads();
    unsigned tid = omp_get_thread_num();
    double time = omp_get_wtime();
    for (int j = 0; j < trials; j++) {
      sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
      for (int i = 0; i < N; i++) {
        sum += a[i];
      }
    }
    time = omp_get_wtime() - time;

    if (tid == 0)
      printf("N: %u # threads: %u time: %e \n", N, threads, time / trials);
  }

  assert(fabs(sum - seq_sum) < 1e-12);

  free(a);
  return 0;
}
