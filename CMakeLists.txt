cmake_minimum_required(VERSION 3.22)
project(xtax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_subdirectory(CLI11)

add_executable(xtax src/main.cpp)
target_link_libraries(xtax PRIVATE Threads::Threads CLI11)
target_compile_features(xtax PRIVATE cxx_std_17)

# ---- Compiler / platform specifics (no user-configurable options) ----
if (MSVC)
  target_compile_options(xtax PRIVATE
    /JMC-              # Disable JustMyCode debugging
    /options:strict
    /MP
    /W4
    /bigobj
    /GF
    $<$<CONFIG:Release>:/O2 /Oi /Ot /Ob3 /GS- /GR- /arch:AVX2>
  )
  target_compile_definitions(xtax PRIVATE
    _CRT_SECURE_NO_WARNINGS=1
    $<$<CONFIG:Release>:_HAS_STATIC_RTTI=0>
  )
  # Static MSVC runtime as in your original
  set_property(TARGET xtax PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
  )
else()
  # GCC/Clang (Linux, macOS, etc.)
  target_compile_options(xtax PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -fno-rtti -fno-stack-protector>
  )

  # Enable AVX2 in Release when the compiler supports the flag
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
  if (COMPILER_SUPPORTS_AVX2)
    target_compile_options(xtax PRIVATE $<$<CONFIG:Release>:-mavx2>)
  endif()
endif()
