cmake_minimum_required(VERSION 3.20)
project(rp_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rp_llm src/main.cpp)
target_include_directories(rp_llm PRIVATE include)

if(MSVC)
    target_compile_options(rp_llm PRIVATE /W4)
    # Let the Visual Studio generator select /Od + /RTC1 for Debug
    # and use /O2 only for Release/RelWithDebInfo/MinSizeRel.
    target_compile_options(rp_llm PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:MinSizeRel>:/O2>
    )
else()
    target_compile_options(rp_llm PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(rp_llm PRIVATE
        $<$<CONFIG:Release>:-O2>
        $<$<CONFIG:RelWithDebInfo>:-O2>
        $<$<CONFIG:MinSizeRel>:-Os>
    )
endif()
