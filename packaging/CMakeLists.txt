set(CPACK_PACKAGE_NAME JointTrackMachineLearning)
set(CPACK_PACKAGE_VENDOR "Gary J Miller Orthopaedic Biomechanics Laboratory")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "An autonomous solutiun to extracting joint kinematics from fluoroscopic images.")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION_MAJOR 1)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_VERBATIM_VARIABLES YES)

set(build_dir_libs "../build/src/Release/")
set(networks_dir "../build/src/Release/Networks")
set(platforms_dir "../build/src/Release/platforms")

file(GLOB MY_DLL_FILES ${build_dir_libs}*.dll)
file(GLOB MY_TRAINED_NETWORKS ${networks_dir}*.pt)
file(GLOB MY_QT_PLATFORMS ${profiles_dir}*.dll)
install(FILES ${MY_DLL_FILES} DESTINATION bin)
install(FILES ${MY_TRAINED_NETWORKS} DESTINATION bin/networks)
install(FILES ${MY_QT_PLATFORMS} DESTINATION bin/platforms)



set(CPACK_GENERATOR NSIS)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(IMPORTED_RUNTIME_ARTIFACTS ${PROJECT_NAME} RUNTIME DESTINATION bin)


include(CPack)
