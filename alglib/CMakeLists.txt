file(GLOB my-files *.cpp *.h)

add_library(alglib STATIC ${my-files})