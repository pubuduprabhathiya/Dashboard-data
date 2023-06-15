#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void foo(double step, int steps, double *sum) {
#pragma nomp for reduce("sum", "+")
  for (unsigned i = 0; i < steps; i++) {
    sum[0] += 4.0 / (1.0 + (i + 0.5) * step * (i + 0.5) * step);
  }
}

int main(int argc, char **argv) {
  unsigned steps = (argc > 1 ? atoi(argv[1]) : 1000000);
  unsigned trials = (argc > 2 ? atoi(argv[2]) : 1000);

  double step = 1 / (double)steps;
  double sum, pi;

#pragma nomp init(argc, argv)
  // Do a warmup run
  for (unsigned i = 0; i < 100; i++)
    foo(step, steps, &sum);

  // Run the pi integration
  clock_t t = clock();
  for (unsigned t = 0; t < trials; t++) {
    foo(step, steps, &sum);
    pi = step * sum;
  }
  t = clock() - t;
  printf("steps: %u pi: %.2f time: %e\n", steps, pi,
         (double)t / (CLOCKS_PER_SEC * trials));

#pragma nomp finalize
  return 0;
}
