from cudaext import PyHeatGrid, cuda_bind_grid, cuda_unbind_grid, cuda_heat_evolve
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
import sys
from icecream import ic


def generate_data(w, h):
	heaters = np.zeros(shape=[h, w], dtype='float32')
	heaters[:10, :10] = 1.0
	heaters[70:170, 70:170] = 1.0
	return heaters


if __name__ == '__main__':
	n_tests = 1000
	w, h = 1024, 1024
	heaters = generate_data(w, h)
	print('Generated heaters')
	grid = PyHeatGrid(heaters)
	print('Initialized grid')
	cuda_bind_grid(grid)
	while True:
		ans = input('Input heat evolution speed (or exit):')
		if ans == 'exit':
			break
		speed = float(ans)
		ans = input('Input number of evolution steps:')
		steps = int(ans)
		cuda_heat_evolve(grid, speed, steps)
		plt.imshow(grid.get_heat_image())
		plt.show()
