#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  unsigned int steps = (argc > 1 ? atoi(argv[1]) : 1000000);
  unsigned int trials = (argc > 2 ? atoi(argv[2]) : 1000);

  double step = 1.0 / (double)steps;
  double x, pi, sum;
  int i;

#pragma omp parallel
  {
    double time = omp_get_wtime();
    unsigned thrds = omp_get_num_threads(), tid = omp_get_thread_num();

    for (int j = 0; j < trials; j++) {
      sum = 0.0;

#pragma omp for reduction(+ : sum) private(x)
      for (i = 0; i < steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
      }

      pi = step * sum;
    }

    if (tid == 0) {
      time = omp_get_wtime() - time;
      printf("N: %u # threads: %u time: %e, pi: %.3f\n", steps, thrds,
             time / trials, pi);
    }
  }

  return 0;
}
