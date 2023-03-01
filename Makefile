# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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
CMAKE_SOURCE_DIR = /home/nicholasverdugo/JTA-CMake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nicholasverdugo/JTA-CMake

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target package
package: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Run CPack packaging tool..."
	/usr/bin/cpack --config ./CPackConfig.cmake
.PHONY : package

# Special rule for the target package
package/fast: package
.PHONY : package/fast

# Special rule for the target package_source
package_source:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Run CPack packaging tool for source..."
	/usr/bin/cpack --config ./CPackSourceConfig.cmake /home/nicholasverdugo/JTA-CMake/CPackSourceConfig.cmake
.PHONY : package_source

# Special rule for the target package_source
package_source/fast: package_source
.PHONY : package_source/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components
.PHONY : list_install_components/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local/fast

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/nicholasverdugo/JTA-CMake/CMakeFiles /home/nicholasverdugo/JTA-CMake//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/nicholasverdugo/JTA-CMake/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named jtml_core

# Build rule for target.
jtml_core: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 jtml_core
.PHONY : jtml_core

# fast build rule for target.
jtml_core/fast:
	$(MAKE) $(MAKESILENT) -f src/core/CMakeFiles/jtml_core.dir/build.make src/core/CMakeFiles/jtml_core.dir/build
.PHONY : jtml_core/fast

#=============================================================================
# Target rules for targets named jtml_core_autogen

# Build rule for target.
jtml_core_autogen: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 jtml_core_autogen
.PHONY : jtml_core_autogen

# fast build rule for target.
jtml_core_autogen/fast:
	$(MAKE) $(MAKESILENT) -f src/core/CMakeFiles/jtml_core_autogen.dir/build.make src/core/CMakeFiles/jtml_core_autogen.dir/build
.PHONY : jtml_core_autogen/fast

#=============================================================================
# Target rules for targets named jtml_nfd

# Build rule for target.
jtml_nfd: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 jtml_nfd
.PHONY : jtml_nfd

# fast build rule for target.
jtml_nfd/fast:
	$(MAKE) $(MAKESILENT) -f src/nfd/CMakeFiles/jtml_nfd.dir/build.make src/nfd/CMakeFiles/jtml_nfd.dir/build
.PHONY : jtml_nfd/fast

#=============================================================================
# Target rules for targets named jtml_nfd_autogen

# Build rule for target.
jtml_nfd_autogen: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 jtml_nfd_autogen
.PHONY : jtml_nfd_autogen

# fast build rule for target.
jtml_nfd_autogen/fast:
	$(MAKE) $(MAKESILENT) -f src/nfd/CMakeFiles/jtml_nfd_autogen.dir/build.make src/nfd/CMakeFiles/jtml_nfd_autogen.dir/build
.PHONY : jtml_nfd_autogen/fast

#=============================================================================
# Target rules for targets named Joint-Track-Machine-Learning

# Build rule for target.
Joint-Track-Machine-Learning: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Joint-Track-Machine-Learning
.PHONY : Joint-Track-Machine-Learning

# fast build rule for target.
Joint-Track-Machine-Learning/fast:
	$(MAKE) $(MAKESILENT) -f src/gui/CMakeFiles/Joint-Track-Machine-Learning.dir/build.make src/gui/CMakeFiles/Joint-Track-Machine-Learning.dir/build
.PHONY : Joint-Track-Machine-Learning/fast

#=============================================================================
# Target rules for targets named Joint-Track-Machine-Learning_autogen

# Build rule for target.
Joint-Track-Machine-Learning_autogen: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Joint-Track-Machine-Learning_autogen
.PHONY : Joint-Track-Machine-Learning_autogen

# fast build rule for target.
Joint-Track-Machine-Learning_autogen/fast:
	$(MAKE) $(MAKESILENT) -f src/gui/CMakeFiles/Joint-Track-Machine-Learning_autogen.dir/build.make src/gui/CMakeFiles/Joint-Track-Machine-Learning_autogen.dir/build
.PHONY : Joint-Track-Machine-Learning_autogen/fast

#=============================================================================
# Target rules for targets named jtml_gpu

# Build rule for target.
jtml_gpu: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 jtml_gpu
.PHONY : jtml_gpu

# fast build rule for target.
jtml_gpu/fast:
	$(MAKE) $(MAKESILENT) -f src/gpu/CMakeFiles/jtml_gpu.dir/build.make src/gpu/CMakeFiles/jtml_gpu.dir/build
.PHONY : jtml_gpu/fast

#=============================================================================
# Target rules for targets named jtml_gpu_autogen

# Build rule for target.
jtml_gpu_autogen: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 jtml_gpu_autogen
.PHONY : jtml_gpu_autogen

# fast build rule for target.
jtml_gpu_autogen/fast:
	$(MAKE) $(MAKESILENT) -f src/gpu/CMakeFiles/jtml_gpu_autogen.dir/build.make src/gpu/CMakeFiles/jtml_gpu_autogen.dir/build
.PHONY : jtml_gpu_autogen/fast

#=============================================================================
# Target rules for targets named JTA_Cost_Functions

# Build rule for target.
JTA_Cost_Functions: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 JTA_Cost_Functions
.PHONY : JTA_Cost_Functions

# fast build rule for target.
JTA_Cost_Functions/fast:
	$(MAKE) $(MAKESILENT) -f src/cost_functions/CMakeFiles/JTA_Cost_Functions.dir/build.make src/cost_functions/CMakeFiles/JTA_Cost_Functions.dir/build
.PHONY : JTA_Cost_Functions/fast

#=============================================================================
# Target rules for targets named JTA_Cost_Functions_autogen

# Build rule for target.
JTA_Cost_Functions_autogen: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 JTA_Cost_Functions_autogen
.PHONY : JTA_Cost_Functions_autogen

# fast build rule for target.
JTA_Cost_Functions_autogen/fast:
	$(MAKE) $(MAKESILENT) -f src/cost_functions/CMakeFiles/JTA_Cost_Functions_autogen.dir/build.make src/cost_functions/CMakeFiles/JTA_Cost_Functions_autogen.dir/build
.PHONY : JTA_Cost_Functions_autogen/fast

#=============================================================================
# Target rules for targets named alglib

# Build rule for target.
alglib: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 alglib
.PHONY : alglib

# fast build rule for target.
alglib/fast:
	$(MAKE) $(MAKESILENT) -f lib/alglib/CMakeFiles/alglib.dir/build.make lib/alglib/CMakeFiles/alglib.dir/build
.PHONY : alglib/fast

#=============================================================================
# Target rules for targets named alglib_autogen

# Build rule for target.
alglib_autogen: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 alglib_autogen
.PHONY : alglib_autogen

# fast build rule for target.
alglib_autogen/fast:
	$(MAKE) $(MAKESILENT) -f lib/alglib/CMakeFiles/alglib_autogen.dir/build.make lib/alglib/CMakeFiles/alglib_autogen.dir/build
.PHONY : alglib_autogen/fast

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... install"
	@echo "... install/local"
	@echo "... install/strip"
	@echo "... list_install_components"
	@echo "... package"
	@echo "... package_source"
	@echo "... rebuild_cache"
	@echo "... JTA_Cost_Functions_autogen"
	@echo "... Joint-Track-Machine-Learning_autogen"
	@echo "... alglib_autogen"
	@echo "... jtml_core_autogen"
	@echo "... jtml_gpu_autogen"
	@echo "... jtml_nfd_autogen"
	@echo "... JTA_Cost_Functions"
	@echo "... Joint-Track-Machine-Learning"
	@echo "... alglib"
	@echo "... jtml_core"
	@echo "... jtml_gpu"
	@echo "... jtml_nfd"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
