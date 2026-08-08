# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 003 U2: single jtml_core target preserved but sources now live in the layered
# dirs (domain/services/coordinator/compute). The header GLOB spans all four
# include dirs so Q_OBJECT headers (optimize_coordinator.h, optimizer_manager.h
# in coordinator/) still get AUTOMOC'd; explicit-impl list stays (GLOB alone
# would miss an impl -> AUTOMOC undefined-symbol link error).
file(GLOB HEADER_FILES
	${PROJECT_SOURCE_DIR}/include/domain/*.h
	${PROJECT_SOURCE_DIR}/include/services/*.h
	${PROJECT_SOURCE_DIR}/include/coordinator/*.h
	${PROJECT_SOURCE_DIR}/include/compute/*.h
)


add_library(jtml_core STATIC
	${PROJECT_SOURCE_DIR}/src/domain/ambiguous_pose_processing.cpp
	${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
	${PROJECT_SOURCE_DIR}/src/domain/direct_data_storage.cpp
	${PROJECT_SOURCE_DIR}/src/domain/direct_optimizer.cpp
	${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
	${PROJECT_SOURCE_DIR}/src/domain/optimize_intent_controller.cpp
	${PROJECT_SOURCE_DIR}/src/domain/pose_file_io.cpp
	${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
	${PROJECT_SOURCE_DIR}/src/domain/sym_trap_functions.cpp
	${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
	${PROJECT_SOURCE_DIR}/src/services/model.cpp
	${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
	${PROJECT_SOURCE_DIR}/src/services/stl_reader.cpp
	${PROJECT_SOURCE_DIR}/src/services/STLReader.cpp
	${PROJECT_SOURCE_DIR}/src/coordinator/optimize_coordinator.cpp
	${PROJECT_SOURCE_DIR}/src/coordinator/optimizer_manager.cpp
	${PROJECT_SOURCE_DIR}/src/compute/curvature_utilities.cpp
	${PROJECT_SOURCE_DIR}/src/compute/frame.cu
	${PROJECT_SOURCE_DIR}/src/compute/machine_learning_tools.cpp
	${HEADER_FILES}
)
# Linking Necessary Libraries
target_link_libraries(jtml_core ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(jtml_core ${TORCH_LIBRARIES})
target_link_libraries(jtml_core ${VTK_LIBRARIES})
target_link_libraries(jtml_core ${OpenCV_LIBRARIES})
target_link_libraries(jtml_core ${CUDA_LIBRARIES})


# Linking In House LIbs and Directories
target_link_libraries(jtml_core ${JTA_LIBS})

# Adding Include Directories
target_include_directories(jtml_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(jtml_core PUBLIC ${CUDA_INCLUDE_DIRS})
#target_include_directories(jtml_core PUBLIC ${CMAKE_BINARY_DIR}/src/gui/${PROJECT_NAME}_autogen/include_Release)
target_include_directories(jtml_core PUBLIC ${PROJECT_SOURCE_DIR}/lib/)
