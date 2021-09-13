del lib\\*.lib
del cudaext*
del wrapper.cpp
rmdir /s /q build

nvcc -I ..\\common -lib -odir lib -o lib/matmul.lib cuda/matmul.cu

python setup.py build_ext -i
