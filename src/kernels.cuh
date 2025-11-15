#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

void gaussianBlurGPU(unsigned char* input, unsigned char* output, int width, int height);
void sobelEdgeGPU(unsigned char* input, unsigned char* output, int width, int height);

#endif
