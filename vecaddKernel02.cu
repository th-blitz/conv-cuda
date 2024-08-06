

// Block dim : 1 and Grid dim : 1
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;  
    int stride = blockDim.x * gridDim.x;  
    
    for(int i = index; i < N; i += stride){
        C[i] = A[i] + B[i];
    }

}
