#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "timer.h"
#include "convKernel.h"
#include <cudnn.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 16

#define OUTPUT_CHANNELS 64
#define INPUT_CHANNELS 3
#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3

#define HEIGHT 1024
#define WIDTH 1024




struct Filter MakeFilters(int32_t output_channels, int32_t input_channels, int32_t filter_height, int32_t filter_width) {

    struct Filter filter;
    filter.output_channels = output_channels;
    filter.input_channels = input_channels;
    filter.height = filter_height;
    filter.width = filter_width;

    filter.elements = (double*)malloc(output_channels * input_channels * filter_height * filter_width * sizeof(double));

    for (int32_t o_channel = 0; o_channel < output_channels; o_channel++) {
        for (int32_t i_channel = 0; i_channel < input_channels; i_channel++) {
            for (int32_t row = 0; row < filter_height; row++) {
                for (int32_t col = 0; col < filter_width; col++) {
                    double element_value = (o_channel + i_channel) * (row + col);
                    filter.elements[
                        o_channel * (input_channels * filter_height * filter_width) +
                        i_channel * (filter_height * filter_width) +
                        row * (filter_width) +
                        col
                    ] = element_value;
                }
            }
        }
    }

    return filter;
}

double* CudnnDeviceFilter(struct Filter filter) {
   
    double* d_cudnn_filter;
    double cudnn_filter [64][3][3][3];

    for (int32_t o_channel = 0; o_channel < filter.output_channels; o_channel++) {
        for (int32_t i_channel = 0; i_channel < filter.input_channels; i_channel++) {
            for (int32_t row = 0; row < filter.height; row++) {
                for (int32_t col = 0; col < filter.width; col++) {
                    cudnn_filter[o_channel][i_channel][row][col] = filter.elements[
                        o_channel * (filter.input_channels * filter.height * filter.width) +
                        i_channel * (filter.height * filter.width) +
                        row * (filter.width) +
                        col
                    ];
                }
            }
        }
    }
    
    size_t size = 64 * 3 * 3 * 3 * sizeof(double); 
    cudaMalloc((void**) &d_cudnn_filter, size);
    cudaMemcpy(d_cudnn_filter, cudnn_filter, size, cudaMemcpyHostToDevice);

    return d_cudnn_filter;
}
void PrintFilter(struct Filter filter, const char* name) {

    int32_t output_channels = filter.output_channels;
    int32_t input_channels = filter.input_channels;
    int32_t height = filter.height;
    int32_t width = filter.width;
    
    printf("\n%s \n",name);
    printf("output_channels : %d \n", output_channels);
    printf("input_channels : %d \n", input_channels);
    printf("height : %d \n", height);
    printf("width : %d \n", width);
    
    for (int32_t o_channel = 0; o_channel < output_channels; o_channel++) {
        printf("\nOutput Channel : %d\n", o_channel);
        for (int32_t i_channel = 0; i_channel < input_channels; i_channel++) {
            printf("--- Input Channel : %d\n", i_channel);
            for (int32_t row = 0; row < height; row++) {
                printf("   | ");
                for (int32_t col = 0; col < width; col++) {
                    double element_value = filter.elements[
                        o_channel * (input_channels * height * width) +
                        i_channel * (height * width) +
                        row * (width) +
                        col
                    ];
                    printf("%lf ", element_value); 
                }
                printf(";\n");
            }
        }
    }
   
}

struct Data_Tensor MakeTensors(int32_t channels, int32_t height, int32_t width, int32_t padding) {
    
    struct Data_Tensor tensor;
    
    int32_t true_height = height + (2 * padding);
    int32_t true_width = width + (2 * padding);

    tensor.channels = channels;
    tensor.height = height;
    tensor.width = width;
    tensor.padding = padding;
    tensor.true_height = true_height;
    tensor.true_width = true_width;
    tensor.elements = (double*)malloc(channels * true_height * true_width * sizeof(double));

    for (int32_t channel = 0; channel < channels; channel++) {
        for (int32_t row = -1 * padding; row < height + padding; row++) {
            for (int32_t col = -1 * padding; col < width + padding; col++) {

                double element = (channel) * (row + col);  
                if (row < 0 || col < 0 || row >= height || col >= width) {
                    element = 0.0;
                }
                tensor.elements[(channel * (true_height * true_width)) + (true_width * (row + padding)) + (col + padding)] = element;

            }
        }
    }
        
    return tensor;
}

void PrintTensor(struct Data_Tensor tensor, const char* name) {

    int32_t channels = tensor.channels;
    int32_t height = tensor.height;
    int32_t width = tensor.width;
    int32_t padding = tensor.padding;
    int32_t true_height = tensor.true_height;
    int32_t true_width = tensor.true_width;

    printf("\n%s \n",name);
    printf("channels : %d \n", channels);
    printf("height : %d \n", height);
    printf("width : %d \n", width);
    printf("padding : %d \n\n", padding);
    
    int32_t row, col;
    for (int32_t channel = 0; channel < channels; channel++) {
        printf("channel : %d\n", channel);
        for (row = -padding; row < height + padding; row++) {
            for (col = -padding; col < width + padding; col++) {

                double element = tensor.elements[(channel * (true_height * true_width)) + (true_width * (row + padding)) + (col + padding)];
                printf("%lf ", element);

            }
            printf(";\n");
        }
    }

}

struct Data_Tensor CopyTensorToDevice(struct Data_Tensor tensor, bool copy) {
    
    struct Data_Tensor device_tensor;

    device_tensor.channels = tensor.channels;
    device_tensor.height = tensor.height;
    device_tensor.width = tensor.width;
    device_tensor.padding = tensor.padding;
    device_tensor.true_height = tensor.true_height;
    device_tensor.true_width = tensor.true_width;
    size_t size = tensor.channels * tensor.true_height * tensor.true_width * sizeof(double); 
    cudaMalloc((void**) &device_tensor.elements, size);
    if (copy)
        cudaMemcpy(device_tensor.elements, tensor.elements, size, cudaMemcpyHostToDevice);
    return device_tensor;
}

struct Filter CopyFilterToDevice(struct Filter filter, bool copy) {
    
    struct Filter device_filter;

    device_filter.input_channels = filter.input_channels;
    device_filter.output_channels = filter.output_channels;
    device_filter.height = filter.height;
    device_filter.width = filter.width;
    size_t size = filter.output_channels * filter.input_channels * filter.height * filter.width * sizeof(double); 
    cudaMalloc((void**)&device_filter.elements, size);
    cudaMemcpy(device_filter.elements, filter.elements, size, cudaMemcpyHostToDevice);
    return device_filter;
}

double PrintSampleResult(struct Data_Tensor input_tensor, struct Data_Tensor output_tensor, struct Filter filter, bool print) {
  
    double checksum = 0.0;

    for (int32_t o_channel = 0; o_channel < output_tensor.channels; o_channel += 1) {
        for (int32_t o_row = 0; o_row < output_tensor.true_height; o_row += 1) {
            for (int32_t o_col = 0; o_col < output_tensor.true_width; o_col += 1) {
                double output_element_value = 0.0;
                for (int32_t i_channel = 0; i_channel < filter.input_channels; i_channel += 1) {
                    for (int32_t row = 0; row < filter.height; row += 1) {
                        for (int32_t col = 0; col < filter.width; col += 1) {
                            output_element_value += filter.elements[
                                o_channel * (filter.input_channels * filter.height * filter.width) +
                                i_channel * (filter.height * filter.width) +
                                (filter.height - 1 - row) * filter.width +
                                (filter.width - 1 - col)
                            ] * input_tensor.elements[
                                i_channel * (input_tensor.true_height * input_tensor.true_width) + 
                                (o_row + row) * (input_tensor.true_width) +
                                (o_col + col) 
                            ];
                        }
                    }
                }
                output_tensor.elements[
                    o_channel * (output_tensor.true_height * output_tensor.true_width) +
                    o_row * (output_tensor.true_width) +
                    o_col
                ] = output_element_value;
                checksum += output_element_value;
            }
        }
    }
    if (print)
        PrintTensor(output_tensor, "print sample output on CPU : "); 
    return checksum;
}

double GenerateCheckSum(struct Data_Tensor output_tensor) {
    double checksum = 0.0;
    for (int32_t o_channel = 0; o_channel < output_tensor.channels; o_channel += 1) {
        for (int32_t o_row = 0; o_row < output_tensor.true_height; o_row += 1) {
            for (int32_t o_col = 0; o_col < output_tensor.true_width; o_col += 1) {
                checksum += output_tensor.elements[
                    o_channel * (output_tensor.true_height * output_tensor.true_width) +
                    o_row * (output_tensor.true_width) +
                    o_col
                ];
            }
        }
    }
    return checksum;
}

void PrintTime(double checksum, double time, uint32_t output_height, uint32_t output_width) {
    double nFlops = (double)( 2 * OUTPUT_CHANNELS * output_height * output_width * INPUT_CHANNELS * FILTER_HEIGHT * FILTER_WIDTH);
    double nFlopsPerSec = nFlops / time;
    double nGFlopsPerSec = nFlopsPerSec*1e-9;
    printf( "Checksum: %lf, Time: %lf (milli sec), GFlopsS: %lf\n",
              checksum, time * 1000, nGFlopsPerSec);
 }

int main(int argc, char* argv[]) {
    
    printf("Number of arguments: %d \n", argc);
    printf("Arguments:\n");
    for (int i = 0; i < argc; i++) {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

   //H = 1024, W = 1024, C = 3, FW = 3, F H = 3, K = 64 
    
//    sscanf(argv[1], "%d", &height);
//    sscanf(argv[1], "%d", &width);
//    sscanf(argv[1], "%d", &input_channels);
//    sscanf(argv[1], "%d", &filter_width);
//    sscanf(argv[1], "%d", &filter_height);
//  
    uint32_t padding = 1; 
    uint32_t input_channels = INPUT_CHANNELS;
    uint32_t output_channels = OUTPUT_CHANNELS;
    uint32_t height = HEIGHT;
    uint32_t width = WIDTH;
    uint32_t filter_height = FILTER_HEIGHT;
    uint32_t filter_width = FILTER_WIDTH;

    uint32_t output_height;
    uint32_t output_width;

    output_height = (
        (height - filter_height + (2*padding)) / 1 // stride = 1 
    ) + 1;

    output_width = (
        (width - filter_width + (2*padding)) / 1 // stride = 1 
    ) + 1;

    dim3 dimGrid(4, 16, 16);
    dim3 dimBlock(4, BLOCK_SIZE, BLOCK_SIZE);

   // printf("Input dim : (%d, %d, %d) \n", input_channels, height, width);
   // printf("Output dim ( dimGrid ) : (%d, %d, %d) \n", output_channels, output_height, output_width);
   // printf("Filter dim ( dimBlock ) : (%d, %d, %d, %d) \n", output_channels, input_channels, filter_height, filter_width);
   
    struct Data_Tensor input_tensor = MakeTensors(input_channels, height, width, padding);
    struct Data_Tensor output_tensor = MakeTensors(output_channels, output_height, output_width, 0);
    struct Filter filter = MakeFilters(output_channels, input_channels, filter_height, filter_width); 
    //PrintTensor(input_tensor, "test input tensor :");
    // PrintFilter(filter, "test filter : "); 

    size_t size = output_tensor.channels * output_tensor.true_height * output_tensor.true_width * sizeof(double); 

    struct Data_Tensor device_input_tensor = CopyTensorToDevice(input_tensor, true);
    struct Data_Tensor device_output_tensor = CopyTensorToDevice(output_tensor, false);
    struct Filter device_filter = CopyFilterToDevice(filter, true);


/////////////////////////   Conv Forward 00 Kernel ////////////////////////////////
    // Invoke kernel for warm up
    ConvForward00<<<dimGrid, dimBlock>>>(device_filter, device_input_tensor, device_output_tensor);      
    // Synchronize to make sure everyone is done in the warmup.
    cudaThreadSynchronize();
    // Set up timer
    initialize_timer();
    start_timer();
    // Invoke kernel for real
    ConvForward00<<<dimGrid, dimBlock>>>(device_filter, device_input_tensor, device_output_tensor);      
    // Synchronize to make sure everyone is done.
    cudaThreadSynchronize() ;
    // Compute and report the timing results
    stop_timer();
    double ConvForward00_time = elapsed_time();
    cudaMemcpy(output_tensor.elements, device_output_tensor.elements, size, cudaMemcpyDeviceToHost);
    // PrintTensor(output_tensor, "test output tensor : ");
    double ConvForward00_checksum = GenerateCheckSum(output_tensor);
    PrintTime(ConvForward00_checksum, ConvForward00_time, output_height, output_width);

/////////////////////////   Conv Forward 01 Kernel ////////////////////////////////
    // Invoke kernel for warm up
    ConvForward01<<<dimGrid, dimBlock>>>(device_filter, device_input_tensor, device_output_tensor);      
    // Synchronize to make sure everyone is done in the warmup.
    cudaThreadSynchronize();
    // Set up timer
    initialize_timer();
    start_timer();
    // Invoke kernel for real
    ConvForward01<<<dimGrid, dimBlock>>>(device_filter, device_input_tensor, device_output_tensor);      
    // Synchronize to make sure everyone is done.
    cudaThreadSynchronize() ;
    // Compute and report the timing results
    stop_timer();
    double ConvForward01_time = elapsed_time();
    cudaMemcpy(output_tensor.elements, device_output_tensor.elements, size, cudaMemcpyDeviceToHost);
    // PrintTensor(output_tensor, "test output tensor : ");
    double ConvForward01_checksum = GenerateCheckSum(output_tensor);
    PrintTime(ConvForward01_checksum, ConvForward01_time, output_height, output_width);


/////////////////////////   Cudnn Kernel ////////////////////////////////

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnStatus_t status;

    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    int NumOfDims_Tensor = 4;
    const int InputTensorDim[] = {1, 3, 1024, 1024};
    const int InputTensorStride[] = {3 * 1024 * 1024, 1024 * 1024, 1024, 1};
    
    const int OutputTensorDim[] = {1, 64, 1024, 1024};
    const int OutputTensorStride[] = {64 * 1024 * 1024, 1024 * 1024, 1024, 1};

    const int FilterTensorDim[] = {64, 3, 3, 3};

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensorNdDescriptor(input_descriptor, CUDNN_DATA_DOUBLE, NumOfDims_Tensor, InputTensorDim, InputTensorStride);
    
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensorNdDescriptor(output_descriptor, CUDNN_DATA_DOUBLE, NumOfDims_Tensor, OutputTensorDim, OutputTensorStride);
    
    
    cudnnCreateFilterDescriptor(&filter_descriptor);
    status = cudnnSetFilterNdDescriptor(filter_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 4, FilterTensorDim);  
    
    if (status != CUDNN_STATUS_SUCCESS) { 
       printf("Error: Failed to create tensor descriptor: %s\n", cudnnGetErrorString(status)); 
    }

    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    status = cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CONVOLUTION,
                                           /*computeType=*/CUDNN_DATA_DOUBLE);
    if (status != CUDNN_STATUS_SUCCESS) { 
       printf("Error: Failed to create tensor descriptor: %s\n", cudnnGetErrorString(status)); 
    }


    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    status = cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        filter_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm);
    if (status != CUDNN_STATUS_SUCCESS) { 
       printf("Error: Failed to create tensor descriptor: %s\n", cudnnGetErrorString(status)); 
    }


    size_t workspace_bytes = 0;
    status = cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                            input_descriptor,
                                            filter_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            convolution_algorithm,
                                            &workspace_bytes);
    if (status != CUDNN_STATUS_SUCCESS) { 
       printf("Error: Failed to create tensor descriptor: %s\n", cudnnGetErrorString(status)); 
    }


    // printf("workspace bytes : %ld\n", workspace_bytes);
    void* device_workspace_bytes;
    cudaMalloc(&device_workspace_bytes, workspace_bytes);

    initialize_timer();
    start_timer();
    
    const double alpha = 1, beta = 0;
    status = cudnnConvolutionForward(cudnn,
                            &alpha,
                            input_descriptor,
                            device_input_tensor.elements,
                            filter_descriptor,
                            device_filter.elements, 
                            convolution_descriptor,
                            convolution_algorithm,
                            device_workspace_bytes,
                            workspace_bytes,
                            &beta,
                            output_descriptor,
                            device_output_tensor.elements);

    cudaThreadSynchronize();
    stop_timer();
    double cudnn_conv_time = elapsed_time();

    if (status != CUDNN_STATUS_SUCCESS) { 
       printf("Error: Failed to create tensor descriptor: %s\n", cudnnGetErrorString(status)); 
    }

    cudaMemcpy(output_tensor.elements, device_output_tensor.elements, size, cudaMemcpyDeviceToHost);
    double cudnn_conv_checksum = GenerateCheckSum(output_tensor);
    PrintTime(cudnn_conv_checksum, cudnn_conv_time, output_height, output_width);
    // PrintTensor(output_tensor, "cudnn output : ");


    double true_checksum = PrintSampleResult(input_tensor, output_tensor, filter, false);
    printf("true checksum : %lf ( calculated on cpu ) \n", true_checksum);

    cudaFree(device_input_tensor.elements);
    cudaFree(device_output_tensor.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_workspace_bytes);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
    
    free(input_tensor.elements);
    free(output_tensor.elements);
    free(filter.elements);


    return 0;

}

    
