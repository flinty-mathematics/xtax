cmake_minimum_required(VERSION 3.22)
project(xtax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_subdirectory(CLI11)

add_executable(xtax src/main.cpp)
add_executable(xbkz src/bkz_main.cpp)
if (WIN32)
  target_sources(xbkz PRIVATE src/xbkz_ui_win32.cpp)
  target_link_libraries(xbkz PRIVATE comctl32)
endif()
add_executable(xdual src/xdual_main.cpp)
add_executable(xsym src/xsym_main.cpp)
add_executable(xweigh src/xweigh_main.cpp)
add_executable(xweigh_lift src/xweigh_lift_main.cpp)
add_executable(xweigh_lift_exact src/xweigh_lift_exact_main.cpp)
add_executable(xweigh_cross src/xweigh_cross_main.cpp)

# All programs share the same dependencies, language level, and build flags.
set(XTAX_TARGETS xtax xbkz xdual xsym xweigh xweigh_lift xweigh_lift_exact
    xweigh_cross)

foreach(tgt IN LISTS XTAX_TARGETS)
  target_link_libraries(${tgt} PRIVATE Threads::Threads CLI11)
  target_compile_features(${tgt} PRIVATE cxx_std_20)
endforeach()

# ---- Optional AVX-512 build path ----
# Opt in with -DXTAX_AVX512=ON to compile the SIMD reductions for AVX-512
# (8-wide __m512d). Default OFF, which keeps the AVX2 code path. This path may
# downclock on some CPUs and its wider summation order can differ in the last
# ULP from the AVX2/scalar paths; cross-platform bit-for-bit determinism is only
# a nice-to-have on this fast path, so that difference is accepted.
option(XTAX_AVX512 "Build the SIMD reductions with AVX-512 (opt-in; may downclock and differ in the last ULP)" OFF)

# ---- Compiler / platform specifics ----
if (MSVC)
  include(CheckCXXCompilerFlag)
  if (XTAX_AVX512)
    set(XTAX_MSVC_ARCH /arch:AVX512)
  else()
    set(XTAX_MSVC_ARCH /arch:AVX2)
  endif()
  foreach(tgt IN LISTS XTAX_TARGETS)
    target_compile_options(${tgt} PRIVATE
      /JMC-              # Disable JustMyCode debugging
      /options:strict
      /MP
      /W4
      /bigobj
      /GF
      # /GL enables whole-program optimization; it pairs with /LTCG at link.
      $<$<CONFIG:Release>:/O2 /Oi /Ot /Ob3 /GS- /GL ${XTAX_MSVC_ARCH}>
    )
    target_link_options(${tgt} PRIVATE $<$<CONFIG:Release>:/LTCG>)
    target_compile_definitions(${tgt} PRIVATE
      _CRT_SECURE_NO_WARNINGS=1
    )
    # Static runtime, RTTI off in Release for a leaner standalone binary.
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:/GR->)
    target_compile_definitions(${tgt} PRIVATE $<$<CONFIG:Release>:_HAS_STATIC_RTTI=0>)
    set_property(TARGET ${tgt} PROPERTY
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endforeach()
else()
  # GCC/Clang (Linux, macOS, etc.)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
  if (XTAX_AVX512)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
  endif()
  foreach(tgt IN LISTS XTAX_TARGETS)
    target_compile_options(${tgt} PRIVATE
      -Wall -Wextra -Wpedantic
      $<$<CONFIG:Release>:-O3 -fno-stack-protector -fno-rtti>
    )
    if (XTAX_AVX512 AND COMPILER_SUPPORTS_AVX512F)
      target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-mavx512f>)
      if (COMPILER_SUPPORTS_AVX512VL)
        target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-mavx512vl>)
      endif()
    elseif (COMPILER_SUPPORTS_AVX2)
      target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-mavx2>)
    endif()
    # Link-time optimization (the GCC/Clang analogue of MSVC /GL + /LTCG).
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-flto>)
    target_link_options(${tgt} PRIVATE $<$<CONFIG:Release>:-flto>)
  endforeach()
endif()

if (XTAX_AVX512)
  message(STATUS "xtax: AVX-512 SIMD path ENABLED (opt-in; may differ in the last ULP)")
else()
  message(STATUS "xtax: AVX-512 SIMD path disabled (set -DXTAX_AVX512=ON to enable; default AVX2)")
endif()

# ---- Optional fast floating point for the FP-heavy reducer / dual paths ----
# Opt in with -DXTAX_FAST_MATH=ON. This lets the compiler reassociate and use
# FMA in the hot floating-point loops, which is faster but breaks the strict,
# portable rounding the default build guarantees (results may differ in the last
# ULP and are no longer bit-for-bit reproducible across builds). Default OFF.
# Applied to xbkz (the reduction core) and xdual (its double-precision dual
# maintenance and Frobenius reductions).
option(XTAX_FAST_MATH "Compile the xbkz reducer and xdual dual loops with fast floating point (opt-in; breaks strict determinism)" OFF)
if (XTAX_FAST_MATH)
  if (MSVC)
    target_compile_options(xbkz PRIVATE $<$<CONFIG:Release>:/fp:fast>)
    target_compile_options(xdual PRIVATE $<$<CONFIG:Release>:/fp:fast>)
    target_compile_options(xsym PRIVATE $<$<CONFIG:Release>:/fp:fast>)
  else()
    target_compile_options(xbkz PRIVATE $<$<CONFIG:Release>:-ffast-math>)
    target_compile_options(xdual PRIVATE $<$<CONFIG:Release>:-ffast-math>)
    target_compile_options(xsym PRIVATE $<$<CONFIG:Release>:-ffast-math>)
  endif()
  message(STATUS "xbkz/xdual/xsym: fast floating point ENABLED (opt-in; last-ULP/determinism differences)")
else()
  message(STATUS "xbkz/xdual/xsym: fast floating point disabled (set -DXTAX_FAST_MATH=ON to enable)")
endif()

# ---- Optional CUDA weighing-matrix annealer ----
option(XTAX_BUILD_CUDA "Build xweigh_cuda when a CUDA compiler is available" ON)
set(XWEIGH_CUDA_ARCHITECTURES "75;86;89" CACHE STRING
    "CUDA architectures for xweigh_cuda")

if (XTAX_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(xweigh_cuda
      src/xweigh_cuda_main.cpp
      src/xweigh_cuda.cu
    )
    target_link_libraries(xweigh_cuda PRIVATE
      Threads::Threads
      CLI11
      CUDA::cudart
      CUDA::curand
    )
    target_compile_features(xweigh_cuda PRIVATE cxx_std_20)
    set_target_properties(xweigh_cuda PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES "${XWEIGH_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION OFF
    )

    if (MSVC)
      target_compile_options(xweigh_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/JMC- /options:strict /W4 /bigobj /GF>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /Oi /Ot /Ob3 /GS->
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20199>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
      )
      target_compile_definitions(xweigh_cuda PRIVATE
        _CRT_SECURE_NO_WARNINGS=1
      )
      set_property(TARGET xweigh_cuda PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
      target_compile_options(xweigh_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -fno-stack-protector>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
      )
    endif()

    message(STATUS "xweigh_cuda: enabled for CUDA architectures ${XWEIGH_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "xweigh_cuda: CUDA compiler not found; target disabled")
  endif()
endif()
