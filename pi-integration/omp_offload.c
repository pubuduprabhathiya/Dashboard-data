#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int steps = (argc > 1 ? atoi(argv[1]) : 1000000);
  int trials = (argc > 2 ? atoi(argv[2]) : 1000);

  float pi;
  double t, sum, step = 1.0 / (double)steps;

  for (int j = 0; j < 100; j++) {
    sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
    for (int i = 0; i < steps; i++) {
      float x = ((double)i + 0.5) * step;
      sum += 4.0 / (1.0 + x * x);
    }
    pi = sum * step;
  }

  t = omp_get_wtime();
  for (int j = 0; j < trials; j++) {
    sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
    for (int i = 0; i < steps; i++) {
      float x = ((double)i + 0.5) * step;
      sum += 4.0 / (1.0 + x * x);
    }
    pi = sum * step;
  }
  t = omp_get_wtime() - t;

  printf("N %d time: %e, pi: %.3f\n", steps, t / trials, pi);

  return 0;
}
