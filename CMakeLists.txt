cmake_minimum_required(VERSION 3.18)
project(FruCom LANGUAGES CXX CUDA) 


set(CMAKE_CUDA_ARCHITECTURES "86;89") 


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")



include_directories(include)


file(GLOB_RECURSE CUDA_SOURCES "src/kernels/*.cu")
set(CUDA_SOURCES
    ${KERNEL_SOURCES}
    src/core/tensor.cu 
    src/kernels/rmsnorm.cu        
)
file(GLOB_RECURSE CPP_SOURCES "src/memory/*.cpp" "src/ops/*.cpp" "src/model/*.cpp" "src/loader/*.cpp" "src/common/*.cpp")


add_library(core_engine STATIC ${CUDA_SOURCES} ${CPP_SOURCES})

add_executable(chat_cli apps/chat.cpp)
target_link_libraries(chat_cli PRIVATE core_engine)