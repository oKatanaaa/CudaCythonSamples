from cudaext import cuda_sum, cpu_sum
import numpy as np
from timeit import timeit
import sys
from icecream import ic


def generate_data(n):
	a = np.ones(n, dtype='float32')
	return a


if __name__ == '__main__':
	n_tests = 1000
	n = 2**21
	a = generate_data(n)
	single_thread = 1
	sixteen_threads = 16
	print(f'Running default test on n={n} elements for n_tests={n_tests} times.')
	ic(timeit('np.sum(a)', globals=globals(), number=n_tests))
	ic(timeit('cpu_sum(a, single_thread)', globals=globals(), number=n_tests))
	ic(timeit('cpu_sum(a, sixteen_threads)', globals=globals(), number=n_tests))
	# The first run is for the initialization routines to be done
	ic(timeit('cuda_sum(a)', globals=globals(), number=n_tests))
	ic(timeit('cuda_sum(a)', globals=globals(), number=n_tests))
	while True:
		ans = input("Input 'exit' to stop the test. Otherwise input number of the elements:")
		if ans == 'exit':
			break
		n = int(ans)
		print(f'n={n}')
		a = generate_data(n)
		ic(timeit('np.sum(a)', globals=globals(), number=n_tests))
		ic(timeit('cpu_sum(a, single_thread)', globals=globals(), number=n_tests))
		ic(timeit('cpu_sum(a, sixteen_threads)', globals=globals(), number=n_tests))
		# The first run is for the initialization routines to be done
		ic(timeit('cuda_sum(a)', globals=globals(), number=n_tests))
		ic(timeit('cuda_sum(a)', globals=globals(), number=n_tests))
