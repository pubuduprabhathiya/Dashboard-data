export uniform float integrate(uniform double step, uniform int steps) {
  double sum = 0;
  int i;
  double x;

  foreach (i = 0 ... steps) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  return reduce_add(sum);
}
