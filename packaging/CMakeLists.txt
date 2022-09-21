set(CPACK_PACKAGE_NAME JointTrackMachineLearning)
set(CPACK_PACKAGE_VENDOR "Gary J Miller Orthopaedic Biomechanics Laboratory")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "An autonomous solutiun to extracting joint kinematics from fluoroscopic images.")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION_MAJOR 1)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_VERBATIM_VARIABLES YES)

set(build_dir_libs ${CMAKE_BINARY_DIR}/src/gui/Release)
set(networks_dir ${build_dir_libs}/Networks)
set(platforms_dir ${build_dir_libs}/platforms)

file(GLOB MY_DLL_FILES ${build_dir_libs}/*.dll)

install(FILES ${MY_DLL_FILES} DESTINATION bin)
install(DIRECTORY ${networks_dir} DESTINATION bin)
install(DIRECTORY ${platforms_dir} DESTINATION bin)

set(CPACK_GENERATOR NSIS)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(IMPORTED_RUNTIME_ARTIFACTS ${PROJECT_NAME} RUNTIME DESTINATION bin)

include(CPack)