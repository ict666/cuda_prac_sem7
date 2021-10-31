#include <cstdio>
#include <iostream>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define SAFE_CALL( CallInstruction ) { \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) { \
         printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		 throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL( KernelCallInstruction ){ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel execution, aborting..."; \
    } \
}

float edge_detection[3][3] = {
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1
};

float identity[3][3] = {
    0, 0, 0,
    0, 1, 0,
    0, 0, 0
};

float sharpen[3][3] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};

float gaussian_blur[5][5] = {
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1
};

__global__ void cuda_filter(unsigned char *img, unsigned char *img_res, float *filter, 
    int filter_size, int height, int width, int img_col_size, int channels) {

    int img_main_coord = blockIdx.x * blockDim.x + threadIdx.x;
    int img_x = img_main_coord % width;
    int img_y = img_main_coord / height;
    unsigned char *p_out = img_res + img_main_coord * channels;

    if (img_x < width && img_x >= 0 && img_y < height && img_y >= 0) {

        float *img_rgba = (float *)malloc(sizeof(*img_rgba) * channels);
        for (int i = 0; i < channels; ++i) {
            img_rgba[i] = 0.;
        }
        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                int img_x_curr = img_x + j - 1;
                int img_y_curr = img_y + i - 1;
                if (img_x_curr >= width) {
                    img_x_curr -= width;
                }
                if (img_x_curr < 0) {
                    img_x_curr += width;
                }
                if (img_y_curr >= height) {
                    img_y_curr -= height;
                }
                if (img_y_curr < 0) {
                    img_y_curr +=height;
                }
                int offset = img_y_curr * img_col_size + img_x_curr * channels;
                for (int k = 0; k < channels; ++k) {
                    img_rgba[k] += (float)*(img + k + offset) * *(filter + filter_size * i + j);
                }
            }
        }
        for (int i = 0; i < channels; ++i) {
            if (i != 3) {
                *(p_out + i) = (unsigned char)max(min((float)255., img_rgba[i]), (float)0.);
            } else {
                *(p_out + i) = 255;
            }
        }
    free(img_rgba);
    }
}



//filter_type 0 - edge detection v1, 1 - edge detection v2, 2 - gaussian blur, image_size (0 - small image, 1 - big image)
int main(int argc, char **argv) {
    int filter_type, image_size;
    sscanf(argv[1], "%d", &filter_type);
    sscanf(argv[2], "%d", &image_size);

    const char *img_path;
    if (image_size == 0) {
        img_path = "images/small.png";
    } else {
        img_path = "images/big.png";
    }

    int filter_size;
    switch (filter_type) {
        case 0:
            filter_size = 3;
            break;
        case 1:
            filter_size = 3;
            break;
        case 2:
            filter_size = 5;
            break;
        default:
            break;
    }
    float *filter = (float *)malloc(filter_size * filter_size * sizeof(*filter));
    switch (filter_type) {
        case 0:
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    *(filter + filter_size * i + j) = edge_detection[i][j];
                }
            }
            break;
        case 1:
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    *(filter + filter_size * i + j) = sharpen[i][j];
                }
            }
            break;
        case 2:
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    *(filter + filter_size * i + j) = gaussian_blur[i][j] / 256.;
                }
            }
            break;
        default:
            break;
    }

    int width, height, channels;
    unsigned char *img = stbi_load(img_path, &width, &height, &channels, 0);
    int img_size = width * height * channels;
    if (img == NULL) {
        printf("Error in loading image\n");
        exit(1);
    }
    printf("Loaded image with\n width:%d\n height:%d\n channels:%d\n\n", width, height, channels);

    int img_res_size = img_size;
    unsigned char *img_res = (unsigned char *)malloc(img_res_size * sizeof(*img_res));
    if (img_res == NULL) {
        printf("Unable to allocate memory to img_res_size\n");
        exit(1);
    }
    int img_col_size = width * channels;
    //загрузка входного изображения на устройство и выделение памяти для результирующего изображения
    unsigned char *device_img, *device_img_res;
    float *device_filter;
    size_t img_byte_size = width * height * channels * sizeof(*img);

    size_t filter_byte_size = filter_size * filter_size * sizeof(*device_filter);
    SAFE_CALL(cudaMalloc(&device_filter, filter_byte_size));
    SAFE_CALL(cudaMalloc(&device_img, img_byte_size));
    SAFE_CALL(cudaMalloc(&device_img_res, img_byte_size));

    float kernel_time = 0, total_time = 0., temp_time;
    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));

    SAFE_CALL(cudaEventRecord(start));
    SAFE_CALL(cudaMemcpy(device_filter, filter, filter_byte_size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_img, img, img_byte_size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    SAFE_CALL(cudaEventElapsedTime(&temp_time, start, stop));
    total_time += temp_time;

    cudaDeviceProp device_prop;
    SAFE_CALL(cudaGetDeviceProperties(&device_prop, 0));
    int threads_in_block = floor(device_prop.maxThreadsPerBlock);
    
    // запуск вычислений на GPU
    dim3 blockDims(threads_in_block, 1, 1);
    dim3 gridDims(ceil((width * height) / (float)threads_in_block), 1, 1);
    std::cout << threads_in_block << std::endl;
    std::cout << ceil(width * height / (float)threads_in_block) << std::endl;

    SAFE_CALL(cudaEventRecord(start));
    SAFE_KERNEL_CALL((cuda_filter<<<gridDims, blockDims>>>(device_img, device_img_res, device_filter, filter_size, height, width, img_col_size, channels)));
    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    SAFE_CALL(cudaEventElapsedTime(&temp_time, start, stop));
    total_time += temp_time;
    kernel_time = temp_time;
    
    //пересылка с девайса на цпу
    SAFE_CALL(cudaEventRecord(start));
    SAFE_CALL(cudaMemcpy(img_res, device_img_res, img_byte_size, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    SAFE_CALL(cudaEventElapsedTime(&temp_time, start, stop));
    total_time += temp_time;

    std::cout << "KERNEL AND COPY TIME: " << total_time << std::endl;
    std::cout << "KERNEL TIME: " << kernel_time << std::endl;

    //сохранение изображения
    const char *img_save_path = "images/output/image.png";
    stbi_write_png(img_save_path, width, height, channels, img_res, width * channels);
    printf("Image written!\n");
    free(img);
    free(img_res);
    free(filter);
    SAFE_CALL(cudaFree(device_img));
    SAFE_CALL(cudaFree(device_img_res));
    SAFE_CALL(cudaFree(device_filter));
    return 0;
}