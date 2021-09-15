Note on the compilation difference compared to other samples:

- In this sample we need to add `-Xcompiler /MD` directive to nvcc compile (see build.bat), otherwise the linking fails.
- The reason of the error is that the CUDA code is compiled with static C Runtime Library, whereas Cython code is 
compiled with dynamic C Runtime Library causing the conflict during linking.
- The conflict is caused by Numpy dependency. If such dependency is removed, the sample builds just fine.
- 