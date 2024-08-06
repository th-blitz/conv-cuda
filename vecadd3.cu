#include <stdio.h>
#include "timer.h"
#include "vecaddKernel.h"


float* array_a;
float* array_b;
float* array_c;


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

    cudaMallocManaged(&array_a, vec_size);
    cudaMallocManaged(&array_b, vec_size);
    cudaMallocManaged(&array_c, vec_size);

    for (int i = 0; i < size; ++i) {
        array_a[i] = (float)i;
        array_b[i] = (float)(size - i);
        array_c[i] = 0.0;
    }


    AddVectors<<< dimGrid, dimBlock >>>(array_a, array_b, array_c, size);
    cudaThreadSynchronize();

    initialize_timer();
    start_timer();

    AddVectors<<< dimGrid, dimBlock >>>(array_a, array_b, array_c, size);
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

    int i;
    for (i = 0; i < size; ++i) {
        float val = array_c[i];
        //printf("%d : %f\n", i, val);
        if (fabs(val - size) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == size) ? "PASSED" : "FAILED");


    cudaFree(array_a);
    cudaFree(array_b);
    cudaFree(array_c);

    return 0;
}

