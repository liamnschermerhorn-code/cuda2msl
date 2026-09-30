# MetalShaderSupport.cmake
# Provides add_metal_library() to compile .metal files into a .metallib

# Find the Metal compiler — discover cryptex path dynamically, fall back to xcrun
# The cryptex mount path changes with each toolchain update, so we search for it
execute_process(
    COMMAND find /private/var/run/com.apple.security.cryptexd/mnt -name "metal" -path "*/usr/bin/metal" -not -path "*/32023/*"
    OUTPUT_VARIABLE METAL_CRYPTEX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    TIMEOUT 5
)
if(METAL_CRYPTEX_PATH)
    # Take the first result if multiple found
    string(REGEX REPLACE "\n.*" "" METAL_CRYPTEX_PATH "${METAL_CRYPTEX_PATH}")
    get_filename_component(METAL_CRYPTEX_DIR "${METAL_CRYPTEX_PATH}" DIRECTORY)
    find_program(METAL_COMPILER metal PATHS "${METAL_CRYPTEX_DIR}" NO_DEFAULT_PATH)
    find_program(METALLIB_TOOL metallib PATHS "${METAL_CRYPTEX_DIR}" NO_DEFAULT_PATH)
endif()

if(NOT METAL_COMPILER)
    execute_process(COMMAND xcrun -sdk macosx -f metal OUTPUT_VARIABLE METAL_COMPILER OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT METAL_COMPILER)
    message(FATAL_ERROR "Metal compiler not found. Install Metal toolchain or set METAL_COMPILER.")
endif()

if(NOT METALLIB_TOOL)
    execute_process(COMMAND xcrun -sdk macosx -f metallib OUTPUT_VARIABLE METALLIB_TOOL OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

message(STATUS "Metal compiler: ${METAL_COMPILER}")
message(STATUS "Metal linker:   ${METALLIB_TOOL}")

function(add_metal_library TARGET)
    cmake_parse_arguments(METAL "" "" "SOURCES" ${ARGN})

    set(AIR_FILES "")

    foreach(SOURCE ${METAL_SOURCES})
        get_filename_component(SOURCE_NAME ${SOURCE} NAME_WE)
        set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_NAME}.air)

        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND ${METAL_COMPILER}
                -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
                -o ${AIR_FILE}
                -std=metal3.0
                -fmodules-cache-path=${CMAKE_BINARY_DIR}/metalcache
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
            COMMENT "Compiling Metal shader ${SOURCE}"
        )
        list(APPEND AIR_FILES ${AIR_FILE})
    endforeach()

    set(METALLIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.metallib)

    add_custom_command(
        OUTPUT ${METALLIB_FILE}
        COMMAND ${METALLIB_TOOL}
            ${AIR_FILES}
            -o ${METALLIB_FILE}
        DEPENDS ${AIR_FILES}
        COMMENT "Linking Metal library ${TARGET}.metallib"
    )

    add_custom_target(${TARGET} ALL DEPENDS ${METALLIB_FILE})

    # Make the metallib path available to consumers
    set(${TARGET}_METALLIB ${METALLIB_FILE} PARENT_SCOPE)
endfunction()
