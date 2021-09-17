import numpy as np


cdef extern  from "cuda/heat_grid.h":
	cdef cppclass HeatGrid:
		int w
		int h
		float totalTime
		# Declare only those member that we will be accessing from Python.
		unsigned char 	*heat_img
		float 			*host_outSrc
		void init_grid(int w, int h);
		void init_heaters(float *heat_grid);
		void destroy_grid();


cdef extern from "cuda/kernel_runner.h":
	void evolve_heat(HeatGrid *grid, float speed, int time_steps);
	void bind_grid(HeatGrid *grid);
	void unbind_grid(HeatGrid *grid);


cdef class PyHeatGrid:
	cdef:
		HeatGrid* grid
		float[:, :] heat_img

	def __cinit__(self, int w, int h):
		self.grid = new HeatGrid()
		self.grid.init_grid(w, h)
		self.heat_img = <float[:h, :w]>self.grid.host_outSrc

	def init_heaters(self, float[:, :] init_heaters):
		self.grid.init_heaters(&init_heaters[0, 0])

	def get_heat_image(self):
		return np.asarray(self.heat_img)

	def totalTime(self):
		return self.grid.totalTime

	def destroy_grid(self):
		self.grid.destroy_grid()


def cuda_bind_grid(PyHeatGrid grid):
	bind_grid(grid.grid)

def cuda_unbind_grid(PyHeatGrid grid):
	bind_grid(grid.grid)

def cuda_heat_evolve(PyHeatGrid grid, float speed, int time_steps):
	evolve_heat(grid.grid, speed, time_steps)

