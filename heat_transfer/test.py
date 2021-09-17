from cudaext import PyHeatGrid, cuda_bind_grid, cuda_unbind_grid, cuda_heat_evolve
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def generate_data(w, h):
	heaters = np.zeros(shape=[h, w], dtype='float32')
	heaters[:10, :10] = 1.0
	heaters[70:170, 70:170] = 1.0
	return heaters


if __name__ == '__main__':
	w, h = 512, 512
	heaters = generate_data(w, h)
	print('Generated heaters')
	grid = PyHeatGrid(w, h)
	print('Initialized grid')
	cuda_bind_grid(grid)
	grid.init_heaters(heaters)
	while True:
		ans = input('Input heat evolution speed (or exit). Recommended speed=0.25.\n')
		if ans == 'exit':
			break
		speed = float(ans)
		ans = input('Input number of evolution steps:')
		steps = int(ans)
		cuda_heat_evolve(grid, speed, steps)
		print(f'Total time is {grid.totalTime()} ms')
		sns.heatmap(grid.get_heat_image())
		plt.show()

	grid.destroy_grid()
