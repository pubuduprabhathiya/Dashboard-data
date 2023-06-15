export uniform float sum_reduction(uniform float input[], uniform int count) {
  float sum = 0;
  foreach (i = 0 ... count)
    sum += input[i];
  return reduce_add(sum);
}
