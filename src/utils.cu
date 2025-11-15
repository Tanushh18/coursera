#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

unsigned char* loadImage(const char* path, int* w, int* h) {
    int channels;
    unsigned char* data = stbi_load(path, w, h, &channels, 1); 
    return data;
}

void saveImage(const char* path, unsigned char* data, int w, int h) {
    stbi_write_jpg(path, w, h, 1, data, 90);
}
