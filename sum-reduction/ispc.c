#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include KERNEL_FILE

int main(int argc, char *argv[]) {
  unsigned N = (argc > 1 ? atoi(argv[1]) : 1 << 10);
  unsigned trials = (argc > 2 ? atoi(argv[2]) : 1000);

  float *input = (float *)malloc(N * sizeof(float));
  float seq_sum = 0, sum;

  for (int i = 0; i < N; i++) {
    input[i] = (float)rand();
    seq_sum += input[i];
  }

  clock_t time = clock();
  for (unsigned t = 0; t < trials; t++)
    sum = sum_reduction(input, N);
  time = clock() - time;

  assert(fabs(sum - seq_sum) < 1e-12);
  printf("N: %u time: %e\n", N, (double)time / (CLOCKS_PER_SEC * trials));

  free(input);
  return 0;
}
