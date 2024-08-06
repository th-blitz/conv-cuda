
#ifndef __MMKERNEL__
#define __MMKERNEL__



struct Data_Tensor {
    int32_t channels;
    int32_t height;
    int32_t width;
    int32_t padding;
    int32_t true_height;
    int32_t true_width;
    double* elements;
};

struct Filter {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t height;
    uint32_t width;
    double* elements;
};



__global__ void ConvForward00(const struct Filter filter, const struct Data_Tensor input_tensor, struct Data_Tensor output_tensor);
__global__ void ConvForward01(const struct Filter filter, const struct Data_Tensor input_tensor, struct Data_Tensor output_tensor);



#endif

