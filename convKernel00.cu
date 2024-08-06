#include <stdint.h>
#include <stdio.h>
#include "convKernel.h"





__global__ void ConvForward(const struct Filter filter, const struct Data_Tensor input_tensor, struct Data_Tensor output_tensor) {
  
    int32_t thread_x = threadIdx.x;
    int32_t thread_y = threadIdx.y;
    int32_t thread_z = threadIdx.z;

    int32_t x_stride = blockDim.x;
    int32_t y_stride = blockDim.y;
    int32_t z_stride = blockDim.z;

    int32_t grid_size_x = output_tensor.channels / gridDim.x;
    int32_t grid_offset_x = blockIdx.x * grid_size_x;

    int32_t grid_size_y = output_tensor.height / gridDim.y;
    int32_t grid_offset_y = blockIdx.y * grid_size_y;
    
    int32_t grid_size_z = output_tensor.width / gridDim.z;
    int32_t grid_offset_z = blockIdx.z * grid_size_z;


    for (int32_t o_channel = thread_x + grid_offset_x; o_channel < grid_offset_x + grid_size_x; o_channel += x_stride) {
        for (int32_t o_row = thread_y + grid_offset_y; o_row < grid_offset_y + grid_size_y; o_row += y_stride) {
            for (int32_t o_col = thread_z + grid_offset_z; o_col < grid_offset_z + grid_size_z; o_col += z_stride) {

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
            }
        }
    }
 
}

//    double output_element_value = 0.0;
//    printf("thread_x %d\n", thread_x);
//    for (int32_t i_channel = 0; i_channel < filter.input_channels; i_channel += 1) {
//        for (int32_t row = 0; row < filter.height; row += 1) {
//            for (int32_t col = 0; col < filter.width; col += 1) {
//                output_element_value += filter.elements[
//                    thread_x * (filter.input_channels * filter.height * filter.width) +
//                    i_channel * (filter.height * filter.width) +
//                    (filter.height - 1 - row) * filter.width +
//                    (filter.width - 1 - col)
//                ] * input_tensor.elements[
//                    i_channel * (input_tensor.true_height * input_tensor.true_width) + 
//                    (thread_y + row) * (input_tensor.true_width) +
//                    (thread_z + col) 
//                ];
//                printf("%f \n", output_element_value);
//            }
//        }
//    }
//    output_tensor.elements[
//        thread_x * (output_tensor.true_height * output_tensor.true_width) +
//        thread_y * (output_tensor.true_width) +
//        thread_z
//    ] = output_element_value;
// 
//}
//    // printf("block x: %d, block y: %d, block z: %d , thread x: %d , thread y: %d \n", block_x, block_y, block_z, thread_x, thread_y);
//    double o_channel_value = 0.0;
//    for (int32_t i_channel = 0; i_channel < filter.input_channels; i_channel += 1) {
//        double i_channel_value = 0.0;
//        double i_value = input_tensor.elements[
//            i_channel * ( input_tensor.true_height * input_tensor.true_width ) +
//            (block_y + thread_x) * ( input_tensor.true_width ) +
//            (block_z + thread_y)
//        ];  
//        double f_value = filter.elements[
//            block_x * ( filter.input_channels * filter.height * filter.width ) +
//            i_channel * ( filter.height * filter.width ) +
//            (filter.height - 1 - thread_x) * (filter.width) +
//            (filter.width - 1 - thread_y)
//        ];
////            printf("block x: %d, i_channel: %d, thread_x: %d, thread_y: %d, f_value: %f\n", block_x, i_channel, thread_x, thread_y, f_value);
////           printf("block x: %d, i_channel: %d, thread_x: %d, thread_y: %d, i_value: %f\n", block_x, i_channel, thread_x, thread_y, i_value);
//        // printf("i_value : %lf, f_value : %lf \n", i_value, f_value);
//        i_channel_value = i_value * f_value;
//        __syncthreads();
//        o_channel_value += i_channel_value;
//    }
//    // printf("block_x: %d, block_y: %d, block_z: %d, value: %f\n", block_x, block_y, block_z, o_channel_value);
//    output_tensor.elements[
//        block_x * ( output_tensor.true_height * output_tensor.true_width ) +
//        block_y * ( output_tensor.true_width ) +
//        block_z
//    ] = o_channel_value;
//
//
//    output_tensor.elements[
//        block_x * ( output_tensor.true_height * output_tensor.true_width ) +
//        block_y * ( output_tensor.true_width ) +
//        block_z
//    ] += input_tensor.elements[
//        thread_x * ( input_tensor.true_height * input_tensor.true_width ) +
//        (block_y + thread_y) * ( input_tensor.true_width ) +
//        (block_z + thread_z)
//    ] * filter.elements[
//        block_x * ( filter.input_channels * filter.height * filter.width ) +
//        thread_x * ( filter.height * filter.width ) +
//        (filter.height - 1 - thread_y) * (filter.width) +
//        (filter.width - 1 - thread_z)
//    ];
//
//


//    for (int32_t o_channel = 0; o_channel < filter.output_channels; o_channel += 1) {
//        for (int32_t row = 0; row < output_tensor.height; row += 1) {
//            for (int32_t col = 0; col < output_tensor.width; col += 1) {
//                output_tensor[
//                    o_channel * ( output_tensor.height * output_tensor.width ) +
//                    row * ( output_tensor.width ) + 
//                    col 
//                ] = output_value; 
//            }
//        }
//    }





