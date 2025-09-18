cmake_minimum_required(VERSION 4.1)
project(xtax CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(/JMC-) # Disable JustMyCode Debugging.
add_compile_options(/options:strict)
add_compile_options(/MP)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

## Warnings ##
add_compile_options(/W4)
add_compile_options(/bigobj) # enable larger object file format
add_compile_options(/GF) # enable read-only string pooling
add_compile_options($<$<CONFIG:Release>:/GR->) # DISABLE RTTI
add_compile_options($<$<CONFIG:Release>:/O2>) # maximum optimizations (favor speed)
add_compile_options($<$<CONFIG:Release>:/Oi>) # enable intrinsic functions
add_compile_options($<$<CONFIG:Release>:/Ot>) # favor code speed
add_compile_options($<$<CONFIG:Release>:/Ob3>) # aggressive inlining
add_compile_options($<$<CONFIG:Release>:/GS->) # Disable security checks
add_compile_options($<$<CONFIG:Release>:/arch:AVX2>)
add_definitions(-D_CRT_SECURE_NO_WARNINGS=0) # Suppress: msvc secure runtime for sprintf etc.

# disable "static" RTTI too (note add_definitions can't take generator expressions)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_HAS_STATIC_RTTI=0")
set(CMAKE_CC_FLAGS_RELEASE "${CMAKE_CC_FLAGS_RELEASE} /D_HAS_STATIC_RTTI=0")

add_subdirectory(CLI11)

add_executable(xtax src/main.cpp)
target_link_libraries(xtax PRIVATE Threads::Threads CLI11)
