#include <stdio.h>
#include "kernels.cuh"

unsigned char* loadImage(const char* path, int* w, int* h);
void saveImage(const char* path, unsigned char* data, int w, int h);

int main() {
    int width, height;

    unsigned char* input = loadImage("../data/input.jpg", &width, &height);

    if (!input) {
        printf("Failed to load input image\n");
        return -1;
    }

    unsigned char* blurred = (unsigned char*)malloc(width * height);
    unsigned char* edges   = (unsigned char*)malloc(width * height);

    printf("Running Gaussian blur...\n");
    gaussianBlurGPU(input, blurred, width, height);
    saveImage("../output/blurred.jpg", blurred, width, height);

    printf("Running Sobel edge detection...\n");
    sobelEdgeGPU(blurred, edges, width, height);
    saveImage("../output/edges.jpg", edges, width, height);

    printf("DONE. Output saved in /output folder.\n");

    free(input);
    free(blurred);
    free(edges);

    return 0;
}
