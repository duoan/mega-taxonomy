cmake_minimum_required(VERSION 3.24)
project(indexflat_routeb LANGUAGES CXX)

option(INDEXFLAT_ENABLE_CUDA "Enable CUDA backend" ON)
option(INDEXFLAT_ENABLE_METAL "Enable Metal backend on macOS" ON)
option(INDEXFLAT_ENABLE_OPENMP "Enable OpenMP threading" ON)
option(INDEXFLAT_ENABLE_TESTS "Build tests" ON)
option(INDEXFLAT_ENABLE_BENCH "Build benchmark CLI" ON)
option(INDEXFLAT_ENABLE_TORCH_IDE_TARGET "Add torch extension IDE compile target" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(indexflat
  src/dispatch.cpp
  src/cpu/indexflat_cpu.cpp
  src/cpu/indexflat_cpu_x86.cpp
  src/cpu/indexflat_cpu_arm.cpp
)

target_include_directories(indexflat PUBLIC include src)
target_compile_options(indexflat PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(indexflat PROPERTIES OUTPUT_NAME indexflat)

add_library(indexflat_c SHARED src/capi.cpp)
target_include_directories(indexflat_c PUBLIC include src)
target_link_libraries(indexflat_c PRIVATE indexflat)
target_compile_options(indexflat_c PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(indexflat_c PROPERTIES OUTPUT_NAME indexflat_c)

if(INDEXFLAT_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(indexflat PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(indexflat PRIVATE INDEXFLAT_HAS_OPENMP=1)
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(indexflat PRIVATE -mavx2 -mfma)
endif()

if(INDEXFLAT_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(indexflat_cuda OBJECT src/cuda/indexflat_cuda.cu)
    target_include_directories(indexflat_cuda PRIVATE include src)
    set_target_properties(indexflat_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set_property(TARGET indexflat_cuda PROPERTY CUDA_ARCHITECTURES 75;80;86;89;90)
    endif()
    target_sources(indexflat PRIVATE $<TARGET_OBJECTS:indexflat_cuda>)
    target_link_libraries(indexflat PRIVATE CUDA::cudart)
    target_compile_definitions(indexflat PRIVATE INDEXFLAT_WITH_CUDA=1)
  else()
    target_sources(indexflat PRIVATE src/cuda/indexflat_cuda_stub.cpp)
  endif()
else()
  target_sources(indexflat PRIVATE src/cuda/indexflat_cuda_stub.cpp)
endif()

if(APPLE AND INDEXFLAT_ENABLE_METAL)
  enable_language(OBJCXX)
  target_sources(indexflat PRIVATE src/metal/indexflat_metal.mm)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(indexflat PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  target_compile_definitions(indexflat PRIVATE INDEXFLAT_WITH_METAL=1)
else()
  target_sources(indexflat PRIVATE src/metal/indexflat_metal_stub.cpp)
endif()

if(INDEXFLAT_ENABLE_TESTS)
  enable_testing()
  add_executable(test_indexflat tests/test_indexflat.cpp)
  target_link_libraries(test_indexflat PRIVATE indexflat)
  add_test(NAME test_indexflat COMMAND test_indexflat)
endif()

if(INDEXFLAT_ENABLE_BENCH)
  add_executable(bench_indexflat bench/bench_indexflat.cpp)
  target_link_libraries(bench_indexflat PRIVATE indexflat)
endif()

if(INDEXFLAT_ENABLE_TORCH_IDE_TARGET)
  set(INDEXFLAT_TORCH_PYTHON "")
  if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(INDEXFLAT_TORCH_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  endif()

  find_package(Python3 COMPONENTS Interpreter)
  if(NOT INDEXFLAT_TORCH_PYTHON AND Python3_Interpreter_FOUND)
    set(INDEXFLAT_TORCH_PYTHON "${Python3_EXECUTABLE}")
  endif()

  if(INDEXFLAT_TORCH_PYTHON)
    execute_process(
      COMMAND "${INDEXFLAT_TORCH_PYTHON}" -c "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
      OUTPUT_VARIABLE INDEXFLAT_TORCH_INCLUDE_PATHS
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND "${INDEXFLAT_TORCH_PYTHON}" -c "import sysconfig; print(sysconfig.get_paths().get('include',''))"
      OUTPUT_VARIABLE INDEXFLAT_PYTHON_INCLUDE_PATH
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

  if(INDEXFLAT_TORCH_INCLUDE_PATHS)
    add_library(indexflat_torch_ide OBJECT EXCLUDE_FROM_ALL mega_taxonomy/indexflat/torch_extension.cpp)
    target_include_directories(indexflat_torch_ide PRIVATE include src ${INDEXFLAT_TORCH_INCLUDE_PATHS})
    if(INDEXFLAT_PYTHON_INCLUDE_PATH)
      target_include_directories(indexflat_torch_ide PRIVATE ${INDEXFLAT_PYTHON_INCLUDE_PATH})
    elseif(Python3_INCLUDE_DIRS)
      target_include_directories(indexflat_torch_ide PRIVATE ${Python3_INCLUDE_DIRS})
    endif()
    target_compile_features(indexflat_torch_ide PRIVATE cxx_std_17)
  endif()
endif()
