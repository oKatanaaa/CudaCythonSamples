cimport cython
from cython.parallel cimport prange, parallel
from libc.stdlib cimport malloc, free


cdef extern from "cuda/kernel.h":
	void gpu_matmul(float *a, float *b, float *out, int a_dim1, int a_dim2, int b_dim2)


def cuda_matmul(float[:, :] a, float[:, :] b, float[:, :] c):
	cdef:
		int a_dim1 = a.shape[0]
		int a_dim2 = a.shape[1]
		int b_dim2 = b.shape[1]
	gpu_matmul(&a[0, 0], &b[0, 0], &c[0, 0], a_dim1, a_dim2, b_dim2)


# Turn off all the smart checks for performance sake
@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def smart_cpu_matmul(float[:, :] a, float[:, :] b, float[:, :] c, int n_threads):
	"""
	Memory optimized dummy implementation of matrix multiply in Cython.
	"""
	cdef:
		int i, j, k, jj
		float sum
		float *buffer


	with nogil, parallel(num_threads=n_threads):
		buffer = <float*>malloc(b.shape[0] * sizeof(float))

		for j in prange(b.shape[1], schedule='static'):
			# Copy column data into the buffer to accelerate access
			# to the corresponding elements.
			for jj in range(b.shape[0]):
				buffer[jj] = b[jj, j]

			for i in range(a.shape[0]):
				sum = 0.0
				for k in range(a.shape[1]):
					sum = sum + a[i, k] * buffer[k]
				c[i, j] = sum
		# Free the buffer
		free(<void*>buffer)


@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_matmul(float[:, :] a, float[:, :] b, float[:, :] c, int n_threads):
	"""
	Dummy implementation of matrix multiply in Cython.
	"""
	cdef:
		int i, j, k
		float sum
	with nogil, parallel(num_threads=n_threads):
		for i in prange(a.shape[0], schedule='static'):
			for j in range(b.shape[1]):
				sum = 0.0
				for k in range(a.shape[1]):
					sum = sum + a[i, k] * b[k, j]
				c[i, j] = sum