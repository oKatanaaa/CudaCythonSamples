## Sample description

This sample contains an example of matrix multiplication using CUBLAS. The matmul implementation reuses
already allocated buffers on GPU if matrices' sizes are the
same. It was designed this way to make performance tests more "practical" since real
application won't allocate memory on a GPU every time data is received.

CUBLAS matmul is being compared to other implementations:

- cpu dummy implementation (see `wrapper.pyx`);
- cpu implementation with optimized memory usage (see `wrapper.pyx`). Columns of the B matrix are
being cached to accelerate access to the corresponding elements. This slight modification results in 3x
performance boost.
- numpy.dot

## Performance research

The time measurements presented in the table below were averaged across 100. Only square matrices were used
during test. The time is measured in milliseconds.

| Mat size | CUBLAS | CPU | np.dot | CPU/CUBLAS | Numpy/CUBLAS |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.220 | 1.973 | __0.082__ | 8.968 | 0.372 |
| 256 | 0.307 | 15.41 | __0.280__ | 50.19 | 0.912 |
| 512 | __0.997__ | 274.6 | 1.480 | 275.4 | 1.484 |
| 1024 | __3.393__ | 5753 | 9.998 | 1695 | 2.947 |
| 2048 | __13.80__ | - | 75.47 | - | 5.469 |

The results suggest that for most cases Numpy will be a "go to" choice. CUDA should be used only in cases when matrices
larger 512 are used and being multiplied frequent enough to affect performance.

