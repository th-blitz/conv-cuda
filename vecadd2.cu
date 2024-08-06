#include <stdio.h>
#include "timer.h"
#include "vecaddKernel.h"


float* host_a;
float* host_b;
float* host_c;
float* device_a;
float* device_b;
float* device_c;


int main(int argc, char* argv[]) {

    //printf("Number of arguments: %d \n", argc);
    //printf("Arguments:\n");
    //for (int i = 0; i < argc; i++) {
    //    printf("argv[%d]: %s\n", i, argv[i]);
    //}
    
    int k;
    int grid_dim = 1;
    int block_dim = 1;

    sscanf(argv[1], "%d", &k);
    sscanf(argv[2], "%d", &grid_dim);
    sscanf(argv[3], "%d", &block_dim);

    int size = k * 1000000;
    int values_per_thread = size / (grid_dim * block_dim);

    printf("Total vector size : %d\n", size);

    dim3 dimGrid(grid_dim);
    dim3 dimBlock(block_dim);

    size_t vec_size = size * sizeof(float);

    host_a = (float*)malloc(vec_size);
    host_b = (float*)malloc(vec_size);
    host_c = (float*)malloc(vec_size);

    cudaMalloc((void**)&device_a, vec_size);
    cudaMalloc((void**)&device_b, vec_size);
    cudaMalloc((void**)&device_c, vec_size);

    for (int i = 0; i < size; ++i) {
        host_a[i] = (float)i;
        host_b[i] = (float)(size - i);
        host_c[i] = 0.0;
    }

    cudaMemcpy(device_a, host_a, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, vec_size, cudaMemcpyHostToDevice);

    AddVectors<<< dimGrid, dimBlock >>>(device_a, device_b, device_c, size);
    cudaThreadSynchronize();

    initialize_timer();
    start_timer();

    AddVectors<<< dimGrid, dimBlock >>>(device_a, device_b, device_c, size);
    cudaThreadSynchronize();

    stop_timer();
    double time = elapsed_time();

    int nFlops = size;
    double nFlopsPerSec = nFlops/time;
    double nGFlopsPerSec = nFlopsPerSec*1e-9;

	// Compute transfer rates.
    int nBytes = 3 * 4 * size; // 2N words in, 1N word out
    double nBytesPerSec = nBytes/time;
    double nGBytesPerSec = nBytesPerSec*1e-9;

	// Report timing data.
    printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
             time, nGFlopsPerSec, nGBytesPerSec);

    cudaMemcpy(host_c, device_c, vec_size, cudaMemcpyDeviceToHost);

    int i;
    for (i = 0; i < size; ++i) {
        float val = host_c[i];
        //printf("%d : %f\n", i, val);
        if (fabs(val - size) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == size) ? "PASSED" : "FAILED");


    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}

