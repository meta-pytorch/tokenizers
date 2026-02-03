# Config file for the tokenizers package
# It defines the following variables:
#  TOKENIZERS_FOUND        - True if the tokenizers library was found
#  TOKENIZERS_INCLUDE_DIRS - Include directories for tokenizers
#  TOKENIZERS_LIBRARIES    - Libraries to link against


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was tokenizers-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
include(GNUInstallDirs)
# Directly include sentencepiece library
set_and_check(TOKENIZERS_LIBDIR "${PACKAGE_PREFIX_DIR}/lib")
if(WIN32)
    set(SENTENCEPIECE_LIBRARY "${TOKENIZERS_LIBDIR}/sentencepiece.lib")
else()
    set(SENTENCEPIECE_LIBRARY "${TOKENIZERS_LIBDIR}/libsentencepiece.a")
endif()
if(NOT EXISTS "${SENTENCEPIECE_LIBRARY}")
  message(
    FATAL_ERROR
      "Could not find sentencepiece library at ${SENTENCEPIECE_LIBRARY}"
  )
endif()

find_dependency(re2 REQUIRED)
find_dependency(absl REQUIRED)

# Include the exported targets file
include("${CMAKE_CURRENT_LIST_DIR}/tokenizers-targets.cmake")

# Set the include directories
set_and_check(TOKENIZERS_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")

# Set the libraries and link options
set(TOKENIZERS_LIBRARIES tokenizers::tokenizers)
set_property(
  TARGET tokenizers::tokenizers
  APPEND
  PROPERTY INTERFACE_LINK_OPTIONS "${TOKENIZERS_LINK_OPTIONS}"
)

# Check if the library was found
check_required_components(tokenizers)
