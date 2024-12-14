#include <iostream>
#include <cmath>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    const char* filename = "input_example.jpg";
    int width, height, channels;
    unsigned char* image = stbi_load(filename, &width, &height, &channels, 0);

    if (!image) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    int num_elements=width*height*channels;
    for (int i=0;i<num_elements;i++){
        image[i]=255-image[i];
    }

    stbi_write_jpg("output_example.jpg",width, height, 3, image, 100);




}