
# Define paths to CUDA installation.
SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB := -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/ 

# Define paths to Cudnn installation. 
CUDNN_INCLUDE_PATH := /usr/include/
CUDNN_LIB_PATH := /usr/lib/x86_64-linux-gnu/
CUDNN_LIB := -L$(CUDNN_LIB_PATH) -I$(CUDNN_INCLUDE_PATH)

OPTIONS   :=  -O3

TAR_FILE_NAME := conv-cuda.tar
EXECS := vecadd01 vecadd00 matmult00 matmult01 vecadd02 vecadd03 conv add_arrays

all:$(EXECS)

clean:
	rm -f $(EXECS) *.o

tar: 
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.py *.cpp *.md

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

# Optimized vecaddKernel00.cu
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

# Optimized vecaddKernel00.cu
vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

matmultKernel00.o : matmultKernel00.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS)

matmult00 : matmult.cu matmultKernel.h matmultKernel00.o timer.o
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

# Optimized matmultKernel00.cu
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

# Optimized matmultKernel00.cu
matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o 
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32

# CUDA vector addition ( without unified memory )
vecaddKernel02.o : vecaddKernel02.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

# CUDA vector addition ( without unified memory )
vecadd02 : vecadd2.cu vecaddKernel.h vecaddKernel02.o timer.o
	${NVCC} $< vecaddKernel02.o -o $@ $(LIB) timer.o $(OPTIONS)

# CUDA vector addition ( with unified memory )
vecadd03 : vecadd3.cu vecaddKernel.h vecaddKernel02.o timer.o
	${NVCC} $< vecaddKernel02.o -o $@ $(LIB) timer.o $(OPTIONS)

# compile convKernels
convKernel.o : convKernel.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

# compile conv kernels and cudnn
conv : conv.cu convKernel.h convKernel.o timer.o
	${NVCC} $< convKernel.o -o $@ -lcudnn $(CUDNN_LIB) $(LIB) timer.o $(OPTIONS)

# cpp program to perform vector additions
add_arrays : add_arrays.cpp 
	g++ add_arrays.cpp -o add_arrays
#
#convKernel01.o : convKernel01.cu
#	${NVCC} $< -c -o $@ $(OPTIONS)
#
#conv01 : conv.cu convKernel.h convKernel01.o timer.o
#	${NVCC} $< convKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)
#
#

