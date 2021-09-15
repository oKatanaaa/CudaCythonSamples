cimport cython
cimport numpy as cnp
from cython.parallel cimport prange


cdef extern from "cuda/kernel.h":
	float dotproduct(float *a, float *b, int n)


def cuda_dotproduct(cnp.ndarray[float, ndim=1] a, cnp.ndarray[float, ndim=1] b):
	cdef int n = a.shape[0]
	return dotproduct(&a[0], &b[0], n)


# Turn off all the smart checks for performance sake
@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_dotproduct(float[:] a, float[:] b, int n_threads):
	cdef:
		int i
		float result = 0.0

	for i in prange(a.shape[0], schedule='static', nogil=True, num_threads=n_threads):

		result += a[i] * b[i]
	return result
