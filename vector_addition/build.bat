del lib\\vecadd.lib
del cudaext*
rmdir /s /q build

nvcc -lib -I ..\\common -O2 -o lib/kernel.lib cuda/kernel.cu

python setup.py build_ext -i
