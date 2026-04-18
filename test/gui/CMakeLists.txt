add_executable(image-loading-service-test
    image_loading_service_test.cpp
    ../../src/gui/image_loading_service.cpp
    ../../include/gui/image_loading_service.h
    ../../src/core/frame.cu
    ../../src/core/model.cpp
    ../../src/core/curvature_utilities.cpp
    ../../src/core/stl_reader.cpp
    ../../src/core/STLReader.cpp
)

target_link_libraries(image-loading-service-test PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(image-loading-service-test PRIVATE ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(image-loading-service-test PRIVATE ${VTK_LIBRARIES})
target_link_libraries(image-loading-service-test PRIVATE ${OpenCV_LIBRARIES})

target_include_directories(image-loading-service-test PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(worker-orchestrator-test
    worker_orchestrator_test.cpp
)

add_executable(settings-service-test
    settings_service_test.cpp
    ../../src/gui/settings_service.cpp
    ../../src/core/optimizer_settings.cpp
    ../../src/core/data_structures_6D.cpp
)

target_link_libraries(worker-orchestrator-test PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(worker-orchestrator-test PRIVATE ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(worker-orchestrator-test PRIVATE ${VTK_LIBRARIES})
target_link_libraries(worker-orchestrator-test PRIVATE ${OpenCV_LIBRARIES})

target_link_libraries(settings-service-test PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(settings-service-test PRIVATE ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(settings-service-test PRIVATE ${VTK_LIBRARIES})
target_link_libraries(settings-service-test PRIVATE ${OpenCV_LIBRARIES})
target_link_libraries(settings-service-test PRIVATE ${JTA_LIBS})

target_include_directories(worker-orchestrator-test PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/gpu
    ${PROJECT_SOURCE_DIR}/include/cost_functions
)

target_include_directories(settings-service-test PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
