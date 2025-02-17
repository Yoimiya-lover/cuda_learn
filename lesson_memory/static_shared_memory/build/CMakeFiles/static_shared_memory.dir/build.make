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
CMAKE_SOURCE_DIR = /home/cuda_learn/lesson_memory/static_shared_memory

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cuda_learn/lesson_memory/static_shared_memory/build

# Include any dependencies generated for this target.
include CMakeFiles/static_shared_memory.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/static_shared_memory.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/static_shared_memory.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/static_shared_memory.dir/flags.make

CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o: CMakeFiles/static_shared_memory.dir/flags.make
CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o: ../static_shared_memory.cu
CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o: CMakeFiles/static_shared_memory.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cuda_learn/lesson_memory/static_shared_memory/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o -MF CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o.d -x cu -c /home/cuda_learn/lesson_memory/static_shared_memory/static_shared_memory.cu -o CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o

CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target static_shared_memory
static_shared_memory_OBJECTS = \
"CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o"

# External object files for target static_shared_memory
static_shared_memory_EXTERNAL_OBJECTS =

static_shared_memory: CMakeFiles/static_shared_memory.dir/static_shared_memory.cu.o
static_shared_memory: CMakeFiles/static_shared_memory.dir/build.make
static_shared_memory: CMakeFiles/static_shared_memory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cuda_learn/lesson_memory/static_shared_memory/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable static_shared_memory"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/static_shared_memory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/static_shared_memory.dir/build: static_shared_memory
.PHONY : CMakeFiles/static_shared_memory.dir/build

CMakeFiles/static_shared_memory.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/static_shared_memory.dir/cmake_clean.cmake
.PHONY : CMakeFiles/static_shared_memory.dir/clean

CMakeFiles/static_shared_memory.dir/depend:
	cd /home/cuda_learn/lesson_memory/static_shared_memory/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cuda_learn/lesson_memory/static_shared_memory /home/cuda_learn/lesson_memory/static_shared_memory /home/cuda_learn/lesson_memory/static_shared_memory/build /home/cuda_learn/lesson_memory/static_shared_memory/build /home/cuda_learn/lesson_memory/static_shared_memory/build/CMakeFiles/static_shared_memory.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/static_shared_memory.dir/depend

