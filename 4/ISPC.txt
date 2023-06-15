#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include KERNEL_FILE

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

  clock_t time = clock();
  for (unsigned t = 0; t < trials; t++)
    matrix_multiplication(a, b, c, height_a, width_a, width_b);
  time = clock() - time;

  for (unsigned i = 0; i < height_a; i++) {
    for (unsigned j = 0; j < width_b; j++) {
      double dot = 0;
      for (unsigned k = 0; k < width_a; k++)
        dot += a[i * width_a + k] * b[k * width_b + j];
      assert(fabs(c[i * width_b + j] - dot) < 1e-8);
    }
  }

  printf("N: %u time: %e\n", N, (double)time / (CLOCKS_PER_SEC * trials));

  free(a), free(b), free(c);
  return 0;
}
