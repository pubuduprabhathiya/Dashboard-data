# Build Instructions
To build the CUDA example, make sure to enable CUDA by updating the `CMakeLists.txt`.
```
option(ENABLE_CUDA "Build Cuda example" ON)
```

```sh
mkdir build; cd build; cmake ..; make; cd -
```

# Run Instructions

You can run the OpenMP example specifying the number of
threads as follows:
```sh
OMP_NUM_THREADS=1 ./build/omp-pi-integration
OMP_NUM_THREADS=2 ./build/omp-pi-integration
OMP_NUM_THREADS=4 ./build/omp-pi-integration
```

OpenCL example can be run simply as follows:
```
./build/ocl-pi-integration
```

You might have to change the OpenCL code and recompile to
use your preferred device on a platform.

CUDA example can be run by:
```
./build/cuda-pi-integration
```

All programs accept problem size and the number of trials as command line input.

