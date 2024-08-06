// vecAddKernel00.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// This Kernel adds two Vectors A and B in C on GPU
// without using coalesced memory access.
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;  // Calculate global index for each thread
    int stride = blockDim.x * gridDim.x;  // Total number of threads

	// contigious memory accesses     
    for(int i = index; i < (N * stride); i += stride){
        C[i] = A[i] + B[i];
    }

    __syncthreads();
}
//
//__global__ void AddVectors(const float* A, const float* B, float* C, int N)
//{
//    int blockStartIndex  = blockIdx.x * blockDim.x * N;
//    int threadStartIndex = blockStartIndex + (threadIdx.x * N);
//    int threadEndIndex   = threadStartIndex + N;
//    int i;
//
//    for( i=threadStartIndex; i<threadEndIndex; ++i ){
//        C[i] = A[i] + B[i];
//    }
//}
