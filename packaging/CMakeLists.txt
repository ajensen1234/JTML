# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

set(CPACK_PACKAGE_NAME JointTrackMachineLearning)
set(CPACK_PACKAGE_VENDOR "Gary J Miller Orthopaedic Biomechanics Laboratory")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "An autonomous solution to extracting joint kinematics from fluoroscopic images.")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION_MAJOR 1)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_VERBATIM_VARIABLES YES)

# Set platform-specific variables
if(WIN32)
    set(build_dir_libs ${CMAKE_BINARY_DIR}/src/gui/Release)
    set(networks_dir ${build_dir_libs}/Networks)
    set(platforms_dir ${build_dir_libs}/platforms)

    # Windows-specific DLL handling
    file(GLOB MY_DLL_FILES ${build_dir_libs}/*.dll)
    install(FILES ${MY_DLL_FILES} DESTINATION bin)

    # Additional Windows-specific installations
    install(DIRECTORY ${networks_dir} DESTINATION bin)
    install(DIRECTORY ${platforms_dir} DESTINATION bin)

    set(CPACK_GENERATOR NSIS)

elseif(UNIX)  # Assumes Linux for simplicity
    # Set Linux-specific variables
    set(build_dir_libs ${CMAKE_BINARY_DIR}/src/gui)
    set(networks_dir ${build_dir_libs}/Networks)
    #set(platforms_dir ${build_dir_libs}/platforms)

    # Linux-specific library handling (adjust this if needed for other Unix-like systems)
    file(GLOB MY_DLL_FILES ${build_dir_libs}/*.so)
    install(FILES ${MY_DLL_FILES} DESTINATION lib)

    # Additional Linux-specific installations
    install(DIRECTORY ${networks_dir} DESTINATION share/${CPACK_PACKAGE_NAME})
    install(DIRECTORY ${platforms_dir} DESTINATION share/${CPACK_PACKAGE_NAME})
    install(TARGETS jtml_gpu jtml_core JTA_Cost_Functions
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

    set(CPACK_GENERATOR DEB)  # Use DEB generator for Debian-based systems, adjust as neeset(CPACK_DEBIAN_PACKAGE_DEPENDS "")ded for other package managers
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
    set(CPACK_PACKAGE_CONTACT "Andrew Jensen")

endif()

# Common installations for both Windows and Linux
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(IMPORTED_RUNTIME_ARTIFACTS ${PROJECT_NAME} RUNTIME DESTINATION bin)

include(CPack)
