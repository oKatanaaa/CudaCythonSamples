del lib\\dotproduct.lib
del cudaext*
rmdir /s /q build

nvcc -lib -I ..\\common -O2 -o lib/dotproduct.lib cuda/dotproduct.cu

python setup.py build_ext -i
