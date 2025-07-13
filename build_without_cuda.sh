#!/bin/bash

git clone https://github.com/ggerganov/llama.cpp || { echo "Failed to clone llama.cpp repository"; exit 1; }

cd llama.cpp || { echo "Failed to change directory to llama.cpp"; exit 1; }

mkdir -p build || { echo "Failed to create build directory"; exit 1; }

cd build || { echo "Failed to change directory to build"; exit 1; }

# llama.cpp has seen few changes recently, so I locked it to a commit from May 2025
git checkout 1d36b3670b285e69e58b9d687c770a2a0a192194

cmake .. \
  -DLLAMA_CUBLAS=OFF || { echo "Failed to configure CMake"; exit 1; }

cmake --build . --config Release -j $(nproc) || { echo "Failed to compile"; exit 1; }