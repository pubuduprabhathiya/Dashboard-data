#define TILE_SIZE 32

export void matrix_multiplication(uniform float a[], uniform float b[],
                                  uniform float c[], uniform int M,
                                  uniform int N, uniform int K) {
  uniform float tile_sums[TILE_SIZE];

  for (uniform unsigned int m = 0; m < M; m++) {
    for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_SIZE) {
      foreach (ki = 0 ... TILE_SIZE)
        tile_sums[ki] = 0.0f;

      for (uniform unsigned int n = 0; n < N; n++) {
        foreach (kt = 0 ... TILE_SIZE)
          tile_sums[kt] += a[m * N + n] * b[n * K + k0 + kt];
      }

      foreach (ki = 0 ... TILE_SIZE)
        c[m * K + k0 + ki] = tile_sums[ki];
    }
  }
}
