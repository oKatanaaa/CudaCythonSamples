del lib\\*.lib
del cudaext*
del wrapper.cpp
rmdir /s /q build

nvcc -I ..\\common -lib -odir lib -o lib/kernel.lib cuda/kernel.cu

python setup.py build_ext -i
