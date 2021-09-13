cimport cython
from cython.parallel cimport prange


cdef extern from "cuda/vecadd.h":
	void vec_add(float *a, float *b, float *out, int n)


def cuda_add(float[:] a, float[:] b, float[:] c):
	cdef int n = a.shape[0]
	vec_add(&a[0], &b[0], &c[0], n)


# Turn off all the smart checks for performance sake
@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_add(float[:] a, float[:] b, float[:] c, int n_threads):
	cdef int i
	for i in prange(a.shape[0], schedule='static', nogil=True, num_threads=n_threads):
		c[i] = a[i] + b[i]
