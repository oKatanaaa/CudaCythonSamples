from cudaext import cuda_raytracing
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
import sys
from icecream import ic


def generate_data(w, h):
	return np.empty(shape=[w, h, 3], dtype='uint8')


if __name__ == '__main__':
	n_tests = 1000
	w, h = 1024, 1024
	img = generate_data(w, h)
	cuda_raytracing(img)
	plt.imshow(img)
	plt.show()
