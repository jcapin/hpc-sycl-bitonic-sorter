#!/bin/bash
set -e

# Load Intel oneAPI environment (this is required to use dpcpp, etc.)
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

echo "Compiling SYCL C++ program..."
dpcpp lab/simple.cpp -o lab/simple.out

echo "Running the compiled program..."
./lab/simple.out
