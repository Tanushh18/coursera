#include "kernels.cuh"
#include <stdio.h>

__global__ void gaussianKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float kernel[5][5] = {
        {1, 4, 6, 4, 1},
        {4,16,24,16, 4},
        {6,24,36,24, 6},
        {4,16,24,16, 4},
        {1, 4, 6, 4, 1}
    };

    float sum = 0.0f;
    float normal = 256.0f;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            sum += kernel[ky+2][kx+2] * input[py * width + px];
        }
    }

    output[y * width + x] = (unsigned char)(sum / normal);
}

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) return;

    int Gx =  
        -input[(y-1)*width + (x-1)] - 2*input[y*width + (x-1)] - input[(y+1)*width + (x-1)] +
         input[(y-1)*width + (x+1)] + 2*input[y*width + (x+1)] + input[(y+1)*width + (x+1)];

    int Gy =
        -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)] +
         input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

    int magnitude = min(255, (int)sqrtf(Gx*Gx + Gy*Gy));

    output[y * width + x] = (unsigned char)magnitude;
}

void gaussianBlurGPU(unsigned char* input, unsigned char* output, int width, int height) {
    unsigned char *d_in, *d_out;
    int size = width * height;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width+15)/16, (height+15)/16);

    gaussianKernel<<<grid, block>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

void sobelEdgeGPU(unsigned char* input, unsigned char* output, int width, int height) {
    unsigned char *d_in, *d_out;
    int size = width * height;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width+15)/16, (height+15)/16);

    sobelKernel<<<grid, block>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
