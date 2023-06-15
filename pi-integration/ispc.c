#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include KERNEL_FILE

int main(int argc, char *argv[]) {
  unsigned int steps = (argc > 1 ? atoi(argv[1]) : 1000000);
  unsigned int trials = (argc > 2 ? atoi(argv[2]) : 1000);

  double step = 1.0 / (double)steps;

  clock_t timer = clock();
  double sum;
  for (unsigned t = 0; t < trials; t++)
    sum = integrate(step, steps);

  double pi = step * sum;
  timer = clock() - timer;
  printf("N: %u time: %e pi: %.3f\n", steps,
         (double)timer / (CLOCKS_PER_SEC * trials), pi);
}
