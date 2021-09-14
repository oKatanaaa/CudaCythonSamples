Note on the compilation difference compared to other samples:

- In this sample we need to add `-Xcompiler /MD` directive to nvcc compile (see build.bat), otherwise the linking fails.
Though I managed  to find a way to resolve the issue (adding that directive), the reason of the linking failure is beyond
my qualification, so I leave it to others to investigate.
- The error is caused by Numpy dependency. If such dependency is removed, the sample builds just fine.