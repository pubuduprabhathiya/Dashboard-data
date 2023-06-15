# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hvpubudu/nomp/examples/matrix-multiplication

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hvpubudu/nomp/examples/matrix-multiplication/build

# Include any dependencies generated for this target.
include CMakeFiles/omp-matrix-mul.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/omp-matrix-mul.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/omp-matrix-mul.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/omp-matrix-mul.dir/flags.make

CMakeFiles/omp-matrix-mul.dir/openmp.c.o: CMakeFiles/omp-matrix-mul.dir/flags.make
CMakeFiles/omp-matrix-mul.dir/openmp.c.o: ../openmp.c
CMakeFiles/omp-matrix-mul.dir/openmp.c.o: CMakeFiles/omp-matrix-mul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hvpubudu/nomp/examples/matrix-multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/omp-matrix-mul.dir/openmp.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/omp-matrix-mul.dir/openmp.c.o -MF CMakeFiles/omp-matrix-mul.dir/openmp.c.o.d -o CMakeFiles/omp-matrix-mul.dir/openmp.c.o -c /home/hvpubudu/nomp/examples/matrix-multiplication/openmp.c

CMakeFiles/omp-matrix-mul.dir/openmp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/omp-matrix-mul.dir/openmp.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hvpubudu/nomp/examples/matrix-multiplication/openmp.c > CMakeFiles/omp-matrix-mul.dir/openmp.c.i

CMakeFiles/omp-matrix-mul.dir/openmp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/omp-matrix-mul.dir/openmp.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hvpubudu/nomp/examples/matrix-multiplication/openmp.c -o CMakeFiles/omp-matrix-mul.dir/openmp.c.s

# Object files for target omp-matrix-mul
omp__matrix__mul_OBJECTS = \
"CMakeFiles/omp-matrix-mul.dir/openmp.c.o"

# External object files for target omp-matrix-mul
omp__matrix__mul_EXTERNAL_OBJECTS =

omp-matrix-mul: CMakeFiles/omp-matrix-mul.dir/openmp.c.o
omp-matrix-mul: CMakeFiles/omp-matrix-mul.dir/build.make
omp-matrix-mul: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
omp-matrix-mul: /usr/lib/x86_64-linux-gnu/libpthread.a
omp-matrix-mul: CMakeFiles/omp-matrix-mul.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hvpubudu/nomp/examples/matrix-multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable omp-matrix-mul"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/omp-matrix-mul.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/omp-matrix-mul.dir/build: omp-matrix-mul
.PHONY : CMakeFiles/omp-matrix-mul.dir/build

CMakeFiles/omp-matrix-mul.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/omp-matrix-mul.dir/cmake_clean.cmake
.PHONY : CMakeFiles/omp-matrix-mul.dir/clean

CMakeFiles/omp-matrix-mul.dir/depend:
	cd /home/hvpubudu/nomp/examples/matrix-multiplication/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hvpubudu/nomp/examples/matrix-multiplication /home/hvpubudu/nomp/examples/matrix-multiplication /home/hvpubudu/nomp/examples/matrix-multiplication/build /home/hvpubudu/nomp/examples/matrix-multiplication/build /home/hvpubudu/nomp/examples/matrix-multiplication/build/CMakeFiles/omp-matrix-mul.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/omp-matrix-mul.dir/depend
