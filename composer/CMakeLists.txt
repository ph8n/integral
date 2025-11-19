cmake_minimum_required(VERSION 3.14)
project(integral_composer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/diff_engine.cpp
    src/safety_check.cpp
)

# Main Executable
add_executable(integral_composer src/main.cpp ${SOURCES})
target_link_libraries(integral_composer PRIVATE nlohmann_json::nlohmann_json)

# Tests
enable_testing()

add_executable(unit_tests
    tests/test_main.cpp
    tests/test_types.cpp
    tests/test_diff_engine.cpp
    tests/test_safety_check.cpp
    ${SOURCES}
)
target_link_libraries(unit_tests
    PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
