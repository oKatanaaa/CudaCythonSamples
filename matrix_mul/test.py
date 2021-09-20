from cudaext import cuda_matmul, cpu_matmul, smart_cpu_matmul
import numpy as np
from timeit import timeit
import sys
from icecream import ic


def generate_data(n):
	a = np.random.randn(n, n).astype(dtype='float32', order='C')
	b = np.random.randn(n, n).astype(dtype='float32', order='C')
	# The output buffer of np.dot must be C-contiguous
	c_np = np.empty((n, n), dtype='float32', order='C')
	c_cuda = np.empty((n, n), dtype='float32', order='F')
	return a, b, c_np, c_cuda


def test(n):
	a, b, c_np, c_cuda = generate_data(n)
	a_cuda, b_cuda = a.astype(dtype='float32', order='F', ), b.astype(dtype='float32', order='F')
	single_thread = 1
	sixteen_threads = 16
	vars = globals()
	vars.update(locals())
	# --- Numpy
	ic(timeit('np.dot(a, b, out=c_np)', globals=vars, number=n_tests) / n_tests)
	# --- CPU dummy
	ic(timeit('cpu_matmul(a, b, c_np, single_thread)', globals=vars, number=n_tests) / n_tests)
	ic(timeit('cpu_matmul(a, b, c_np, sixteen_threads)', globals=vars, number=n_tests) / n_tests)
	# --- CPU smart
	ic(timeit('smart_cpu_matmul(a, b, c_np, single_thread)', globals=vars, number=n_tests) / n_tests)
	ic(timeit('smart_cpu_matmul(a, b, c_np, sixteen_threads)', globals=vars, number=n_tests) / n_tests)
	# --- CUDA
	ic(timeit('cuda_matmul(a_cuda, b_cuda, c_cuda)', globals=vars, number=n_tests) / n_tests)
	ic(timeit('cuda_matmul(a_cuda, b_cuda, c_cuda)', globals=vars, number=n_tests) / n_tests)

	if n <= 4:
		print('Check the outputs...')
		np.dot(a, b, out=c_np)
		ic(c_np)
		# For better print formatting
		c_cpu = c_np
		cpu_matmul(a, b, c_cpu, sixteen_threads)
		ic(c_cpu)

		cuda_matmul(a_cuda, b_cuda, c_cuda)
		ic(c_cuda)


if __name__ == '__main__':
	n_tests = 100

	while True:
		ans = input("Input 'exit' to stop the test. Otherwise input number of the elements:")
		if ans == 'exit':
			break
		n = int(ans)
		print(f'n={n}')
		test(n)


