## Description

This repository contains solutions for the university CUDA course.
The solutions contain code samples with Cython + CUDA showing how to 
generate CUDA capable python extensions.

The repository is organized as follows:
- vector_addiction (Lab1)
  - The most basic example of CUDA.
- matrix_mul (Lab2)
  - Uses CUBLAS for matrix multiplication.
- dotproduct
  - Uses shared memory for partial reduction.
- raytracing
  - Uses constant memory for accelerating the access to a list of objects that never changes.

## Tested hardware

All the code was built and tested with the following setup:
- OS: Windows 10.0.19041
- CPU: Ryzen 7 4800H
- GPU: RTX2060 6GB
- CUDA Version: 11.2
- GPU Driver Version: 462.80
- Compiler: Microsoft Visual Studio Community 2019

## Requirements