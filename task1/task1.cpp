#include <cstdio>
#include <iostream>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"




float filter[3][3] = {
    1./9, 1./9, 1./9,
    1./9, 1./9, 1./9,
    1./9, 1./9, 1./9
};

float filter2[3][3] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};

float filter3[3][3] = {
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1
};

//filter_type image_size
int main(int argc, char **argv) {
    int filter_type, image_size;
    sscanf(argv[1], "%d", &filter_type);
    sscanf(argv[2], "%d", &image_size);

    const char *img_path = "images/small/small.png";

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

    int img_x = 0; //column number
    int img_y = 0;  //row number
    int img_col_size = width * channels;
    int img_row_size = height * channels;
    int filter_size = 3;
    float *img_rgba = (float *)malloc(channels * sizeof(*img_rgba));

    for (unsigned char *p_in = img, *p_out = img_res; p_in != img + img_size; p_in += channels, p_out += channels) {
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
                    img_rgba[k] += (float)*(img + k + offset) * filter[i][j];
                }
            }
        }
        for (int i = 0; i < channels; ++i) {
            if (i != 3) {
                *(p_out + i) = (unsigned char)std::max(std::min(255., (double)img_rgba[i]), 0.);
            } else {
                *(p_out + i) = 255;
            }
        }
        img_x++;
        if (img_x == width) {
            img_x = 0;
            img_y++;
            if (img_y == height) {
                break;
            }
        }
    }
    free(img_rgba);

    const char *img_save_path = "images/output/small.png";
    stbi_write_png(img_save_path, width, height, channels, img_res, width * channels);
    free(img);
    free(img_res);
    return 0;
}