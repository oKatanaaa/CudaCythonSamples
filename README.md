## Description

This repository contains solutions for the university CUDA course.
The solutions contain code samples with Cython + CUDA showing how to 
generate CUDA capable python extensions.

The repository is organized as follows:
- **vector_addiction**
  - A simple CUDA program that adds two vectors.
      - The most basic example of CUDA.
- **matrix_mul** (Lab2)
  - A simple CUBLAS program that multiplies two square matrices.
    - Uses CUBLAS for matrix multiplication.
- **dotproduct**
  - An implementation of dot product (or scalar product) in CUDA.
    - Uses shared memory for partial reduction.
- **raytracing**
  - A very simple implementation of raytracing with randomly generated spheres. No lighting, reflections, etc.
    - Uses constant memory for accelerating access to a list of objects that never changes.
- **heat_transfer**
  - A simple (physically inaccurate) example of heat transfer in a grid.
    - Uses texture memory for accelerating access to spatially neighboring pixels.

Yes, every code sample here is something simple!

## How to run the project

Every code sample folder the following structure:
- `cuda` folder - contains all the CUDA code;
- `lib` folder - will contain `.lib` files with compiled CUDA code;
- `build.bat` - script containing instructions for building the sample (compilation, linking);
- `clear.bat` - a helper script to remove all files produced after building;
- `setup.py` - contains instructions for the Cython compiler on how to make the Python extension with CUDA;
- `test.py` - contains code for testing the CUDA extension;
- `wrapper.pyx` - Cython wrapper code around the CUDA code.

To run the sample, complete the following steps:
1. Make sure you have `CUDAHOME` **environment variable** which contains path to the CUDA Toolkit folder. 
   Example: *C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2*.
2. Make sure you have installed python packages listed in `requirements.txt`.
3. Run `build.bat`.
4. Run `test.py`.

## Tested hardware

All the code was built and tested with the following setup:
- OS: Windows 10.0.19041
- CPU: Ryzen 7 4800H
- GPU: RTX2060 6GB
- CUDA Version: 11.2
- GPU Driver Version: 462.80
- Compiler: Microsoft Visual Studio Community 2019
