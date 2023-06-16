#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

// Cuda Functions:
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if(result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Rendering Kernel
__global__
void render(float* frameBuffer, int max_x, int max_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if((i >= max_x) || (j >= max_y)) {
        return;
    }

    int pixel_index = 3 * (j * max_x + i);
    frameBuffer[pixel_index + 0] = float(i) / max_x;
    frameBuffer[pixel_index + 1] = float(j) / max_y;
    frameBuffer[pixel_index + 2] = 0.25f;
}

int main() {
    // Image Parameters
    const auto aspect_ratio = 1.0f;
    const int image_width = 256;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int channels = 3;
    int num_pixels = image_width * image_height;
    size_t frameBuffer_size = 3 * num_pixels * sizeof(float);

    // Allocate the Frame Buffer and set grid parameters
    float *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBuffer_size));
    int numThreadsX = 8;
    int numThreadsY = 8;
    dim3 blocks(image_width / numThreadsX + 1, image_height / numThreadsY + 1);
    dim3 threads(numThreadsX, numThreadsY);

    // Render the buffer
    render<<<blocks, threads>>>(frameBuffer, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned char* image{ new unsigned char[image_width * image_height * channels]{} };

    int idx = 0;
    for(int j = image_height-1; j >= 0; j--) {
        std::cerr << "\rScanlines Remaining: " << j << ' ' << std::flush;
        for(int i = 0; i < image_width; i++) {
            size_t pixel_index = 3 * (j * image_width + i);
            float r = frameBuffer[pixel_index + 0];
            float g = frameBuffer[pixel_index + 1];
            float b = frameBuffer[pixel_index + 2];

            image[idx] = (unsigned char)(255.999f * r);
            image[idx + 1] = (unsigned char)(255.999 * g);
            image[idx + 2] = (unsigned char)(255.999f * b);
            idx += 3;
        }
    }
    
    // Write the pixel array to file, and free the Frame Buffer
    stbi_write_png("images\\colorGradient.png", 
                    image_width, image_height, channels, image, image_width * channels);
    checkCudaErrors(cudaFree(frameBuffer));
    std::cerr << "\nDone!\n";
}
