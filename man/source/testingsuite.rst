Testing Suite
=================

dcEmb uses Googles GTest framework to run it's tests. There is still some work
to be done on these tests, but they are currently split up into three 
executables, roughly corresponding to tests that take a short, moderate, and 
long amount of time to run respectively.


Current tests
##############

run_tests
+++++++++++
A suite of unit tests, covering most base functionality

run_serialization_tests
+++++++++++++++++++++++++
A test suite that runs short versions of the COVID and 3Body models, and 
verifies that the resulting objects can be serialized/deserialized.

run_tests_long
++++++++++++++++
A test suite that runs the full COVID model and verifies it's accuracy against
a known benchmark. May take several hours to run, depending on the platform.


GTest
#######
GTest is available on `GitHub <https://github.com/google/googletest>`_ with 
corresponding documentation on 
`GitHub pages <https://google.github.io/googletest/>`_.