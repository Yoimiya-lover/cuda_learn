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
CMAKE_SOURCE_DIR = //home/cuda_learn/lesson6_GPU_cache

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = //home/cuda_learn/lesson6_GPU_cache/build

# Include any dependencies generated for this target.
include CMakeFiles/GPU_cache.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GPU_cache.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GPU_cache.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GPU_cache.dir/flags.make

CMakeFiles/GPU_cache.dir/GPU_cache.cu.o: CMakeFiles/GPU_cache.dir/flags.make
CMakeFiles/GPU_cache.dir/GPU_cache.cu.o: ../GPU_cache.cu
CMakeFiles/GPU_cache.dir/GPU_cache.cu.o: CMakeFiles/GPU_cache.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=//home/cuda_learn/lesson6_GPU_cache/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/GPU_cache.dir/GPU_cache.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/GPU_cache.dir/GPU_cache.cu.o -MF CMakeFiles/GPU_cache.dir/GPU_cache.cu.o.d -x cu -c //home/cuda_learn/lesson6_GPU_cache/GPU_cache.cu -o CMakeFiles/GPU_cache.dir/GPU_cache.cu.o

CMakeFiles/GPU_cache.dir/GPU_cache.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/GPU_cache.dir/GPU_cache.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/GPU_cache.dir/GPU_cache.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/GPU_cache.dir/GPU_cache.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target GPU_cache
GPU_cache_OBJECTS = \
"CMakeFiles/GPU_cache.dir/GPU_cache.cu.o"

# External object files for target GPU_cache
GPU_cache_EXTERNAL_OBJECTS =

GPU_cache: CMakeFiles/GPU_cache.dir/GPU_cache.cu.o
GPU_cache: CMakeFiles/GPU_cache.dir/build.make
GPU_cache: CMakeFiles/GPU_cache.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=//home/cuda_learn/lesson6_GPU_cache/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable GPU_cache"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GPU_cache.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GPU_cache.dir/build: GPU_cache
.PHONY : CMakeFiles/GPU_cache.dir/build

CMakeFiles/GPU_cache.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GPU_cache.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GPU_cache.dir/clean

CMakeFiles/GPU_cache.dir/depend:
	cd //home/cuda_learn/lesson6_GPU_cache/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" //home/cuda_learn/lesson6_GPU_cache //home/cuda_learn/lesson6_GPU_cache //home/cuda_learn/lesson6_GPU_cache/build //home/cuda_learn/lesson6_GPU_cache/build //home/cuda_learn/lesson6_GPU_cache/build/CMakeFiles/GPU_cache.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GPU_cache.dir/depend

