# Build Instructions

```sh
mkdir build; cd build; cmake ..; make; cd -
```

# Run Instructions

You can run the OpenMP example specifying the number of
threads as follows:
```sh
OMP_NUM_THREADS=1 ./build/omp-matrix-mul
OMP_NUM_THREADS=2 ./build/omp-matrix-mul
OMP_NUM_THREADS=4 ./build/omp-matrix-mul
```

OpenCL example can be run simply as follows:
```
./build/ocl-matrix-mul
```

Both programs accept height of first matrix, width of first matrix,
width of second matrix and the number of trials as command line input.

You might have to change the OpenCL code and recompile to
use your preferred device on a platform.
