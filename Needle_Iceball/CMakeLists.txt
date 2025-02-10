cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Needle_Iceball)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/Needle_Iceball")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "John Doe (AnyWare Corp.)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/Needle_Iceball.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/Needle_Iceball/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Needle_Iceball2)
add_subdirectory(CalculateOverlap)
add_subdirectory(CalculateAIBruteForce)
add_subdirectory(CalculateAIBruteForce3)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
