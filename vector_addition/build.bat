del lib\\vecadd.lib
del cudaext*
rmdir /s /q build

nvcc -lib -I ..\\common -O2 -o lib/vecadd.lib cuda/vecadd.cu

python setup.py build_ext -i
