#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int main() {
    // Image Parameters
    const auto aspect_ratio = 3.0f / 2.0f;
    const int image_width = 300;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int channels = 3;

    unsigned char* image{ new unsigned char[image_width * image_height * channels]{} };

    int idx = 0;
    for(int j = image_height-1; j >= 0; j--) {
        std::cerr << "\rScanlines Remaining: " << j << ' ' << std::flush;
        for(int i = 0; i < image_width; i++) {
            auto r = double(i) / (image_width - 1);
            auto g = double(j) / (image_height - 1);
            auto b = 0.25;

            image[idx] = (unsigned char)(255.999 * r);
            image[idx + 1] = (unsigned char)(255.999 * g);
            image[idx + 2] = (unsigned char)(255.999 * b);
            idx += 3;
        }
    }

    stbi_write_png("images\\colorGradient.png", 
                    image_width, image_height, channels, image, image_width * channels);
    std::cerr << "\nDone!\n";
}
