cimport numpy as cnp


cdef extern from "cuda/kernel.h":
	void trace_spheres(unsigned char *img, int w, int h)


def cuda_raytracing(cnp.ndarray[unsigned char, ndim=3] img):
	cdef int h = img.shape[0], w = img.shape[1]
	return trace_spheres(&img[0, 0, 0], w, h)

