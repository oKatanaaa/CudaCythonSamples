from cudaext import cuda_matmul, cpu_matmul
import numpy as np
from timeit import timeit
import sys
from icecream import ic


def generate_data(n):
	a = np.ones((n, n), dtype='float32', order='C')
	b = np.ones((n, n), dtype='float32', order='C')
	# The output buffer of np.dot must be C-contiguous
	c_np = np.ones((n, n), dtype='float32', order='C')
	c_cuda = np.ones((n, n), dtype='float32', order='C')
	return a, b, c_np, c_cuda


if __name__ == '__main__':
	n_tests = 100
	n = 128
	a, b, c_np, c_cuda = generate_data(n)
	single_thread = 1
	sixteen_threads = 16
	print(f'Running default test on n={n} elements for n_tests={n_tests} times.')
	ic(timeit('np.dot(a, b, out=c_np)', globals=globals(), number=n_tests))
	ic(timeit('cpu_matmul(a, b, c_np, single_thread)', globals=globals(), number=n_tests))
	ic(timeit('cpu_matmul(a, b, c_np, sixteen_threads)', globals=globals(), number=n_tests))
	# The first run is for the initialization routines to be done
	ic(timeit('cuda_matmul(a, b, c_cuda)', globals=globals(), number=n_tests))
	ic(timeit('cuda_matmul(a, b, c_cuda)', globals=globals(), number=n_tests))
	while True:
		ans = input("Input 'exit' to stop the test. Otherwise input number of the elements:")
		if ans == 'exit':
			break
		n = int(ans)
		print(f'n={n}')
		a, b, c_np, c_cuda = generate_data(n)
		ic(timeit('np.dot(a, b, out=c_np)', globals=globals(), number=n_tests))
		ic(timeit('cpu_matmul(a, b, c_np, single_thread)', globals=globals(), number=n_tests))
		ic(timeit('cpu_matmul(a, b, c_np, sixteen_threads)', globals=globals(), number=n_tests))
		ic(timeit('cuda_matmul(a, b, c_cuda)', globals=globals(), number=n_tests))
		ic(timeit('cuda_matmul(a, b, c_cuda)', globals=globals(), number=n_tests))
