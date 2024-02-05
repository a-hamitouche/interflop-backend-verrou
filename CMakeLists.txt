project("interflop_verrou")

set(VR_COMPILE_DEFINITIONS
  "VERROU_NUM_AVG=1"
  "VERROU_DET_HASH=vr_double_tabulation_hash"
  "USE_VERROU_FMA"
  "USE_VERROU_SQRT"
)

set (INTERFLOP_VERROU_SRC
  "interflop_verrou.cxx"
)
set (INTERFLOP_VVERROU_SRC
    "x86_64/interflop_vector_verrou.cxx"
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/x86_64")

add_library(interflop_verrou_base   OBJECT ${INTERFLOP_VERROU_SRC})
target_compile_definitions(interflop_verrou_base PRIVATE  ${CRT_COMPILE_DEFINITIONS} ${VR_COMPILE_DEFINITIONS} 
  "SCALAR" "VECT128" "VECT256" "VECT512")
target_compile_options (interflop_verrou_base PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-mfma" "-mavx2")

add_library(interflop_verrou_scalar OBJECT ${INTERFLOP_VVERROU_SRC})
target_compile_definitions(interflop_verrou_scalar PRIVATE  ${CRT_COMPILE_DEFINITIONS} ${VR_COMPILE_DEFINITIONS} "SCALAR")
target_compile_options (interflop_verrou_scalar PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-mfma")

add_library(interflop_verrou_sse    OBJECT ${INTERFLOP_VVERROU_SRC})
target_compile_definitions(interflop_verrou_sse PRIVATE  ${CRT_COMPILE_DEFINITIONS} ${VR_COMPILE_DEFINITIONS} "VECT128")
target_compile_options (interflop_verrou_sse PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-msse4.2")

add_library(interflop_verrou_avx    OBJECT ${INTERFLOP_VVERROU_SRC})
target_compile_definitions(interflop_verrou_avx PRIVATE  ${CRT_COMPILE_DEFINITIONS} ${VR_COMPILE_DEFINITIONS} "VECT256")
target_compile_options (interflop_verrou_avx PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-mavx2" "-mfma")

add_library(interflop_verrou_avx512    OBJECT ${INTERFLOP_VVERROU_SRC})
target_compile_definitions(interflop_verrou_avx512 PRIVATE  ${CRT_COMPILE_DEFINITIONS} ${VR_COMPILE_DEFINITIONS} "VECT512")
target_compile_options (interflop_verrou_avx512 PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-mavx512f" "-mfma")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
add_library (interflop_verrou SHARED $<TARGET_OBJECTS:interflop_verrou_base>
                                     $<TARGET_OBJECTS:interflop_verrou_scalar>
                                     $<TARGET_OBJECTS:interflop_verrou_sse>
                                     $<TARGET_OBJECTS:interflop_verrou_avx>
                                     $<TARGET_OBJECTS:interflop_verrou_avx512>
)
target_link_options (interflop_verrou PRIVATE ${CRT_LINK_OPTIONS})
target_link_libraries (interflop_verrou ${CRT_LINK_LIBRARIES} interflop_stdlib)