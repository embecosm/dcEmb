# Copyright (c) 2017-2018: Radovan Bast, Roberto Di Remigio,
# and other contributors:

# https://github.com/dev-cafe/cmake-cookbook/contributors

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

find_package(PythonInterp REQUIRED)
find_package(Sphinx REQUIRED)

function(add_sphinx_doc)
  set(options)
  set(oneValueArgs
    SOURCE_DIR
    BUILD_DIR
    CACHE_DIR
    HTML_DIR
    CONF_FILE
    TARGET_NAME
    COMMENT
    )
  set(multiValueArgs)

  cmake_parse_arguments(SPHINX_DOC
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
    )

  configure_file(
    ${SPHINX_DOC_CONF_FILE}
    ${SPHINX_DOC_BUILD_DIR}/conf.py
    @ONLY
    )

  add_custom_target(${SPHINX_DOC_TARGET_NAME}
    COMMAND
      ${SPHINX_EXECUTABLE}
         -q
         -b html
         -c ${SPHINX_DOC_BUILD_DIR}
         -d ${SPHINX_DOC_CACHE_DIR}
         ${SPHINX_DOC_SOURCE_DIR}
         ${SPHINX_DOC_HTML_DIR}
    COMMENT
      "Building ${SPHINX_DOC_COMMENT} with Sphinx"
    VERBATIM
    )

  message(STATUS "Added ${SPHINX_DOC_TARGET_NAME} [Sphinx] target to build documentation")
endfunction()