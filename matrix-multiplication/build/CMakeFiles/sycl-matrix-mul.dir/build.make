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
include CMakeFiles/sycl-matrix-mul.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sycl-matrix-mul.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sycl-matrix-mul.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sycl-matrix-mul.dir/flags.make

CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o: CMakeFiles/sycl-matrix-mul.dir/flags.make
CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o: ../sycl.cpp
CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o: CMakeFiles/sycl-matrix-mul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hvpubudu/nomp/examples/matrix-multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o"
	/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o -MF CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o.d -o CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o -c /home/hvpubudu/nomp/examples/matrix-multiplication/sycl.cpp

CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.i"
	/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hvpubudu/nomp/examples/matrix-multiplication/sycl.cpp > CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.i

CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.s"
	/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hvpubudu/nomp/examples/matrix-multiplication/sycl.cpp -o CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.s

# Object files for target sycl-matrix-mul
sycl__matrix__mul_OBJECTS = \
"CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o"

# External object files for target sycl-matrix-mul
sycl__matrix__mul_EXTERNAL_OBJECTS =

sycl-matrix-mul: CMakeFiles/sycl-matrix-mul.dir/sycl.cpp.o
sycl-matrix-mul: CMakeFiles/sycl-matrix-mul.dir/build.make
sycl-matrix-mul: CMakeFiles/sycl-matrix-mul.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hvpubudu/nomp/examples/matrix-multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sycl-matrix-mul"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sycl-matrix-mul.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sycl-matrix-mul.dir/build: sycl-matrix-mul
.PHONY : CMakeFiles/sycl-matrix-mul.dir/build

CMakeFiles/sycl-matrix-mul.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sycl-matrix-mul.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sycl-matrix-mul.dir/clean

CMakeFiles/sycl-matrix-mul.dir/depend:
	cd /home/hvpubudu/nomp/examples/matrix-multiplication/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hvpubudu/nomp/examples/matrix-multiplication /home/hvpubudu/nomp/examples/matrix-multiplication /home/hvpubudu/nomp/examples/matrix-multiplication/build /home/hvpubudu/nomp/examples/matrix-multiplication/build /home/hvpubudu/nomp/examples/matrix-multiplication/build/CMakeFiles/sycl-matrix-mul.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sycl-matrix-mul.dir/depend
