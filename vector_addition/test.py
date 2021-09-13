from cudaext import cuda_add, cpu_add
import numpy as np
from timeit import timeit
import sys
from icecream import ic


def generate_data(n):
	a = np.ones(n, dtype='float32')
	b = np.ones(n, dtype='float32')
	c = np.ones(n, dtype='float32')
	return a, b, c


if __name__ == '__main__':
	n_tests = 1000
	n = 1024
	a, b, c = generate_data(n)
	single_thread = 1
	sixteen_threads = 16
	print(f'Running default test on n={n} elements for n_tests={n_tests} times.')
	ic(timeit('np.add(a, b, out=c)', globals=globals(), number=n_tests))
	ic(timeit('cpu_add(a, b, c, single_thread)', globals=globals(), number=n_tests))
	ic(timeit('cpu_add(a, b, c, sixteen_threads)', globals=globals(), number=n_tests))
	# The first run is for the initialization routines to be done
	ic(timeit('cuda_add(a, b, c)', globals=globals(), number=n_tests))
	ic(timeit('cuda_add(a, b, c)', globals=globals(), number=n_tests))
	while True:
		ans = input("Input 'exit' to stop the test. Otherwise input number of the elements:")
		if ans == 'exit':
			break
		n = int(ans)
		print(f'n={n}')
		a, b, c = generate_data(n)
		ic(timeit('np.add(a, b, out=c)', globals=globals(), number=n_tests))
		ic(timeit('cpu_add(a, b, c, single_thread)', globals=globals(), number=n_tests))
		ic(timeit('cpu_add(a, b, c, sixteen_threads)', globals=globals(), number=n_tests))
		ic(timeit('cuda_add(a, b, c)', globals=globals(), number=n_tests))
		ic(timeit('cuda_add(a, b, c)', globals=globals(), number=n_tests))
