#include "matmultKernel.h"
#include <stdio.h>




void print_sub_mat(float* sub, int size) {
    for ( int x = 0; x < size; x++ ) {
        for ( int y = 0; y < size; y++) {
            printf("%f ", sub[x * size + y]);
        }
        printf("\n");
    }
}


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

	float *Asub, *Bsub, *Csub;
	
	int thread_row = threadIdx.y;
	int thread_col = threadIdx.x;
	int block_row = blockIdx.y;
	int block_col = blockIdx.x;
	
	Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

	// To store the 4 output values:
	float Cvalues[] = {0, 0, 0, 0, 0, 0, 0, 0};
	// Offsets arrays to offset thread indices while performing matrix multiplications over 32x32 sub matrices.
	int a_row_offset[] = {0, 0, 0, 0, 16, 16, 16, 16};
	int a_col_offset[] = {0, 16, 0, 16, 0, 16, 0, 16};
	int b_row_offset[] = {0, 16, 0, 16, 0, 16, 0, 16};
	int b_col_offset[] = {0, 0, 16, 16, 0, 0, 16, 16};
	
	for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m) {

		Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
		Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];
	
		// Each thread calculates outputs for 4 values across 32x32 sub matrices.
#pragma unroll
    	for (int i = 0; i < 8; i++) {
			
			// load a 16x16 block to shared memory
    		__shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
			__shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
			
			shared_A[thread_row][thread_col] = Asub[(thread_row + a_row_offset[i]) * A.stride + (thread_col + a_col_offset[i])];
			shared_B[thread_row][thread_col] = Bsub[(thread_row + b_row_offset[i]) * B.stride + (thread_col + b_col_offset[i])];
			
			__syncthreads();
			
			// perform matrix multiplications over 16x16 blocks
#pragma unroll
			for(int e=0; e<BLOCK_SIZE; ++e) {
			   Cvalues[i] += shared_A[thread_row][e] * shared_B[e][thread_col];
			} 
			
			__syncthreads();
		
    	}
	}

	// assign all 4 output values:
	Csub[thread_row * C.stride + thread_col] = Cvalues[0] + Cvalues[1];
	Csub[thread_row * C.stride + (thread_col + 16)] = Cvalues[2] + Cvalues[3];
	Csub[(thread_row + 16) * C.stride + thread_col] = Cvalues[4] + Cvalues[5];
	Csub[(thread_row + 16) * C.stride + (thread_col + 16)] = Cvalues[6] + Cvalues[7];
}

