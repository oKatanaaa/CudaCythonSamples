del lib\\*.lib
del cudaext*
rmdir /s /q build

nvcc -lib -Xcompiler /MD -I ../common -O0 -o lib/kernel.lib cuda/kernel_runner.cu cuda/heat_grid.cu

python setup.py build_ext -i
