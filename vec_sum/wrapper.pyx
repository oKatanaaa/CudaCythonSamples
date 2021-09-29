cimport cython
from cython.parallel cimport prange


cdef extern from "cuda/kernel.h":
	float vec_sum(float *a, int n)


def cuda_sum(float[:] a):
	cdef int n = a.shape[0]
	return vec_sum(&a[0], n)


# Turn off all the smart checks for performance sake
@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_sum(float[:] a, int n_threads):
	cdef:
		int i
		float s = 0.0
	for i in prange(a.shape[0], schedule='static', nogil=True, num_threads=n_threads):
		s += a[i]
	return s
