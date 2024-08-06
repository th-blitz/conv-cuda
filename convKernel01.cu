
// copyright (c) th-blitz (https://github.com/th-blitz) 2024

#include <stdint.h>
#include <stdio.h>
#include "convKernel.h"

#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3
#define INPUT_CHANNELS 3 
#define OUTPUT_CHANNELS 64
#define BLOCK_SIZE 16

__global__ void ConvForward(const struct Filter filter, const struct Data_Tensor input_tensor, struct Data_Tensor output_tensor) {
  
    int32_t thread_x = threadIdx.x;
    int32_t thread_y = threadIdx.y;
    int32_t thread_z = threadIdx.z;

    int32_t x_stride = blockDim.x;
    int32_t y_stride = blockDim.y;
    int32_t z_stride = blockDim.z;

    __shared__ double FilterTile[OUTPUT_CHANNELS][INPUT_CHANNELS][FILTER_HEIGHT][FILTER_WIDTH];
    __shared__ double InputTile[INPUT_CHANNELS][BLOCK_SIZE + FILTER_HEIGHT - 1][BLOCK_SIZE + FILTER_WIDTH - 1];

    int32_t grid_size_x = output_tensor.channels / gridDim.x;
    int32_t grid_offset_x = blockIdx.x * grid_size_x;

    int32_t grid_size_y = output_tensor.height / gridDim.y;
    int32_t grid_offset_y = blockIdx.y * grid_size_y;
    
    int32_t grid_size_z = output_tensor.width / gridDim.z;
    int32_t grid_offset_z = blockIdx.z * grid_size_z;



    for (int32_t o_channel = thread_x + grid_offset_x; o_channel < grid_offset_x + grid_size_x; o_channel += x_stride) {
        for (int32_t o_row = thread_y + grid_offset_y; o_row < grid_offset_y + grid_size_y; o_row += y_stride) {
            for (int32_t o_col = thread_z + grid_offset_z; o_col < grid_offset_z + grid_size_z; o_col += z_stride) {

                if (thread_y < 3 && thread_z < 3) {
                    #pragma unroll
                    for (int32_t i_channel = 0; i_channel < INPUT_CHANNELS; i_channel += 1) {
                        FilterTile[o_channel][i_channel][thread_y][thread_z] = filter.elements[
                            o_channel * (filter.input_channels * filter.height * filter.width) +
                            i_channel * (filter.height * filter.width) +
                            (filter.height - 1 - thread_y) * filter.width +
                            (filter.width - 1 - thread_z)
                        ];
                    }
                }

                double* input_val;
                if (thread_x < INPUT_CHANNELS) {

                    input_val = &input_tensor.elements[
                        thread_x * (input_tensor.true_height * input_tensor.true_width) + 
                        (o_row) * (input_tensor.true_width) +
                        (o_col) 
                    ]; 

                    InputTile[thread_x][thread_y][thread_z] = input_val[0]; 

                    if (thread_y == BLOCK_SIZE - 1) {
                        InputTile[thread_x][thread_y + 1][thread_z] = input_val[input_tensor.true_width];                       
                        InputTile[thread_x][thread_y + 2][thread_z] = input_val[2 * input_tensor.true_width];                       
                    }

                    if (thread_z == BLOCK_SIZE - 1) {
                        InputTile[thread_x][thread_y][thread_z + 1] = input_val[1];                       
                        InputTile[thread_x][thread_y][thread_z + 2] = input_val[2];                       
                    }

                    if (thread_y == (BLOCK_SIZE - 1) && thread_z == (BLOCK_SIZE - 1)) {
                        InputTile[thread_x][thread_y + 1][thread_z + 1] = input_val[input_tensor.true_width + 1];                       
                        InputTile[thread_x][thread_y + 1][thread_z + 2] = input_val[input_tensor.true_width + 2]; 
                        InputTile[thread_x][thread_y + 2][thread_z + 1] = input_val[2 * input_tensor.true_width + 1];                       
                        InputTile[thread_x][thread_y + 2][thread_z + 2] = input_val[2 * input_tensor.true_width + 2];                       
                    } 
                }
                                      
                __syncthreads();

                double output_element_value = 0.0;

                for (int32_t i_channel = 0; i_channel < INPUT_CHANNELS; i_channel += 1) {
                    for (int32_t row = 0; row < FILTER_HEIGHT; row += 1) {
                        for (int32_t col = 0; col < FILTER_WIDTH; col += 1) {

                            output_element_value += FilterTile[o_channel][i_channel][row][col] * 
                                InputTile[i_channel][thread_y + row][thread_z + col];

                        }
                    }
                }

                output_tensor.elements[
                        o_channel * (output_tensor.true_height * output_tensor.true_width) +
                        o_row * (output_tensor.true_width) +
                        o_col 
                ] = output_element_value;

                __syncthreads();
            }
        }
    }
 
}
