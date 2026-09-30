#!/bin/bash
# Build transpiler and run test
set -e
c++ -std=c++20 -c src/transpiler/codegen.cpp -I src/transpiler -o build-transpiler/CMakeFiles/cuda2msl_lib.dir/codegen.cpp.o
ar rcs build-transpiler/libcuda2msl_lib.a \
  build-transpiler/CMakeFiles/cuda2msl_lib.dir/codegen.cpp.o \
  build-transpiler/CMakeFiles/cuda2msl_lib.dir/transpiler.cpp.o \
  build-transpiler/CMakeFiles/cuda2msl_lib.dir/parser.cpp.o \
  build-transpiler/CMakeFiles/cuda2msl_lib.dir/mapper.cpp.o \
  build-transpiler/CMakeFiles/cuda2msl_lib.dir/lexer.cpp.o
c++ -std=c++20 build-transpiler/CMakeFiles/cuda2msl.dir/main.cpp.o -L build-transpiler -lcuda2msl_lib -o build-transpiler/cuda2msl
echo "BUILD OK"

PASS=0
TOTAL=0
FAILS=""
for f in site/libraries/pytorch/aten/src/ATen/native/cuda/*.cu; do
  TOTAL=$((TOTAL+1))
  base=$(basename "$f" .cu)
  ./build-transpiler/cuda2msl "$f" -o /tmp/pytorch-metal/${base}.metal 2>/dev/null
  if xcrun metal -c /tmp/pytorch-metal/${base}.metal -o /dev/null 2>/dev/null; then
    PASS=$((PASS+1))
  else
    FAILS="$FAILS $base"
  fi
done
echo "PASS=$PASS / TOTAL=$TOTAL"
echo "FAILS:$FAILS"
