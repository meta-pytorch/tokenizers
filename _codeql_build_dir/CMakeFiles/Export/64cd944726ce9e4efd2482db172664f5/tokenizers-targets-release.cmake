#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tokenizers::tokenizers" for configuration "Release"
set_property(TARGET tokenizers::tokenizers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tokenizers::tokenizers PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtokenizers.a"
  )

list(APPEND _cmake_import_check_targets tokenizers::tokenizers )
list(APPEND _cmake_import_check_files_for_tokenizers::tokenizers "${_IMPORT_PREFIX}/lib/libtokenizers.a" )

# Import target "tokenizers::re2" for configuration "Release"
set_property(TARGET tokenizers::re2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tokenizers::re2 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libre2.a"
  )

list(APPEND _cmake_import_check_targets tokenizers::re2 )
list(APPEND _cmake_import_check_files_for_tokenizers::re2 "${_IMPORT_PREFIX}/lib/libre2.a" )

# Import target "tokenizers::sentencepiece-static" for configuration "Release"
set_property(TARGET tokenizers::sentencepiece-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tokenizers::sentencepiece-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsentencepiece.a"
  )

list(APPEND _cmake_import_check_targets tokenizers::sentencepiece-static )
list(APPEND _cmake_import_check_files_for_tokenizers::sentencepiece-static "${_IMPORT_PREFIX}/lib/libsentencepiece.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
