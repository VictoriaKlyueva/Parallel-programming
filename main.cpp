#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <map>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include <immintrin.h>

using namespace std;

const int threadsNum = 10;

struct Images {
    vector<const char*> imagesNames = {"different_sizes/300x300.png", "different_sizes/400x400.png",
                                       "different_sizes/500x500.png", "different_sizes/600x600.png",
                                       "different_sizes/950x950.png", "different_sizes/2400x2400.png"
    };
    vector <unsigned char*> imagesArray;
    vector <int> sizes;
    vector <int> channels;
};

float gaussFunction(float x, float y, float sigma) {
    return exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
}

float* generatingKernel(float sigma, int kernelSize=20) {
    float* kernel = new float[kernelSize * kernelSize];

    float sum = 0;
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i * kernelSize + j] = gaussFunction(i - kernelSize / 2, j - kernelSize / 2, sigma);
            sum += kernel[i * kernelSize + j];
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

char kernelCounting(unsigned char* image, int width, int height, int channels,
                     float* kernel, int channel, int w, int h, int kernelSize=20) {
    float onePixel = 0;
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            if (((w + i) - kernelSize / 2 >= 0 && (w + i) - kernelSize / 2 < width) &&
                ((h + j) - kernelSize / 2 >= 0 && (h + j) - kernelSize / 2 < height))
                onePixel += kernel[i * kernelSize + j] *
                            image[channels * (((h + j) - kernelSize / 2) * width +
                                              ((w + i) - kernelSize / 2)) + channel];
        }
    }
    return char(onePixel);
}

unsigned char* gaussBlur(unsigned char* image, int width, int height, int channels, float sigma, int kernelSize=20) {
    unsigned char* newImage = new unsigned char[width * height * channels];
    float* kernel = generatingKernel(sigma);

    for (int channel = 0; channel < channels; channel++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                newImage[(h * width + w) * channels + channel] =
                        kernelCounting(image, width, height, channels, kernel, channel, w, h);
            }
        }
    }
    return newImage;
}

unsigned char* openMPGaussBlur(unsigned char* image, int width, int height, int channels, float sigma, int kernelSize=20) {
    unsigned char *newImage = new unsigned char[width * height * channels];
    float* kernel = generatingKernel(sigma);

    int channel, h, w, i, j;
    float onePixel;

    omp_set_num_threads(threadsNum);

#pragma omp parallel for shared(onePixel, image, newImage, width, height, channels, sigma, kernel, kernelSize) private(channel, h, w, i, j)
    for (int channel = 0; channel < channels; channel++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // instead kernel counting function
                onePixel = 0;
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        if (((w + i) - kernelSize / 2 >= 0 && (w + i) - kernelSize / 2 < width) &&
                            ((h + j) - kernelSize / 2 >= 0 && (h + j) - kernelSize / 2 < height))
                            onePixel += kernel[i * kernelSize + j] *
                                        image[channels * (((h + j) - kernelSize / 2) * width +
                                        ((w + i) - kernelSize / 2)) + channel];
                    }
                }
                newImage[(h * width + w) * channels + channel] = onePixel;
            }
        }
    }
    return newImage;
}

unsigned char* __attribute__ ((__target__ ("avx2"))) vectorizedGaussBlur(unsigned char* image, int width, int height, int channels, float sigma, int kernelSize=20) {
    unsigned char* newImage = new unsigned char[width * height * channels];
    float* kernel = generatingKernel(sigma);

    for (int channel = 0; channel < channels; channel++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w += 8) { // gap
                __m256 onePixel = _mm256_setzero_ps();
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        if (((w + i - kernelSize / 2) >= 0 && (w + i - kernelSize / 2) < width)
                        && ((h + j - kernelSize / 2) >= 0 && (h + j - kernelSize / 2) < height)) {
                            int currentIndex = ((h + j - kernelSize / 2) * width + (w + i - kernelSize / 2)) * channels + channel;
                            __m256 getKernel = _mm256_set1_ps(kernel[i * kernelSize + j]);
                            __m256 getImage = _mm256_setr_ps(image[currentIndex], image[currentIndex + 3], image[currentIndex + 6],
                                                             image[currentIndex + 9], image[currentIndex + 12], image[currentIndex + 15],
                                                             image[currentIndex + 18], image[currentIndex + 21]);
                            onePixel = _mm256_add_ps(_mm256_mul_ps(getKernel, getImage), onePixel);
                        }
                    }
                }

                float* currentImage = (float*)&onePixel;
                int count = 0;
                for (int k = 0; k < 8; k++) {
                    if (w + k < width) {
                        newImage[((h * width + w) * channels + channel) + count] = (unsigned char)currentImage[k];
                        count += 3;
                    }
                }

            }
        }
    }
    return newImage;
}

unsigned char* mosaicFilter(unsigned char* image, int width, int height, int channels, int blockSize) {
    unsigned char* mosaicImage = new unsigned char[width * height * channels];

    for (int i = 0; i < height; i += blockSize) {
        for (int j = 0; j < width; j += blockSize) {
            int red = 0; int green = 0; int blue = 0;

            for (int x = i; x < min(i + blockSize, height); x++) {
                for (int y = j; y < min(j + blockSize, width); y++) {
                    int currentIndex = (x * width + y) * channels;
                    red += image[currentIndex];
                    green += image[currentIndex + 1];
                    blue += image[currentIndex + 2];
                }
            }

            // find mean color value
            red /= (blockSize * blockSize);
            green /= (blockSize * blockSize);
            blue /= (blockSize * blockSize);


            for (int x = i; x < min(i + blockSize, height); x++) {
                for (int y = j; y < min(j + blockSize, width); y++) {
                    mosaicImage[(x * width + y) * channels] = red;
                    mosaicImage[(x * width + y) * channels + 1] = green;
                    mosaicImage[(x * width + y) * channels + 2] = blue;
                }
            }
        }
    }

    return mosaicImage;
}

unsigned char* openMPMosaicFilter(unsigned char* image, int width, int height, int channels, int blockSize) {
    unsigned char* mosaicImage = new unsigned char[width * height * channels];

    int red, green, blue;
    int i, j, x, y;

#pragma omp parallel for shared(red, green, blue, image, mosaicImage, width, height, channels, blockSize) private(i, j, x, y)
    for (int i = 0; i < height; i += blockSize) {
        for (int j = 0; j < width; j += blockSize) {
            red = 0; blue = 0; green = 0;

            for (int x = i; x < min(i + blockSize, height); x++) {
                for (int y = j; y < min(j + blockSize, width); y++) {
                    red += image[(x * width + y) * channels];
                    green += image[(x * width + y) * channels + 1];
                    blue += image[(x * width + y) * channels + 2];
                }
            }

            // find mean color value
            red /= (blockSize * blockSize);
            green /= (blockSize * blockSize);
            blue /= (blockSize * blockSize);


            for (int x = i; x < min(i + blockSize, height); x++) {
                for (int y = j; y < min(j + blockSize, width); y++) {
                    mosaicImage[(x * width + y) * channels] = red;
                    mosaicImage[(x * width + y) * channels + 1] = green;
                    mosaicImage[(x * width + y) * channels + 2] = blue;
                }
            }
        }
    }

    return mosaicImage;
}

unsigned char* __attribute__ ((__target__ ("avx2"))) vectorizedMosaicFilter(unsigned char* image, int width, int height, int channels, int blockSize) {
    unsigned char* mosaicImage = new unsigned char[width * height * channels];

    // blocks
    for (int i = 0; i < height; i += blockSize) {
        for (int j = 0; j < width; j += blockSize) {
            __m256 red = _mm256_setzero_ps();
            __m256 green = _mm256_setzero_ps();
            __m256 blue = _mm256_setzero_ps();

            // pixels in block
            for (int x = i; x < min(i + blockSize, height); x++) {
                for (int y = j; y < min(j + blockSize, width); y+=8) {
                    int currentIndex = (x * width + y) * channels;
                    __m256 currentRed = _mm256_setr_ps(image[currentIndex], image[currentIndex + 3], image[currentIndex + 6],
                                                       image[currentIndex + 9], image[currentIndex + 12], image[currentIndex + 15],
                                                       image[currentIndex + 18], image[currentIndex + 21]);
                    red = _mm256_add_ps(currentRed, red);

                    __m256 currentGreen = _mm256_setr_ps(image[currentIndex + 1], image[currentIndex + 4], image[currentIndex + 7],
                                                         image[currentIndex + 10], image[currentIndex + 13], image[currentIndex + 16],
                                                         image[currentIndex + 19], image[currentIndex + 22]);
                    green = _mm256_add_ps(currentGreen, green);

                    __m256 currentBlue = _mm256_setr_ps(image[currentIndex + 2], image[currentIndex + 5], image[currentIndex + 8],
                                                        image[currentIndex + 11], image[currentIndex + 14], image[currentIndex + 17],
                                                        image[currentIndex + 20], image[currentIndex + 23]);
                    blue = _mm256_add_ps(currentBlue, blue);
                }
            }

            // find mean color value
            red = _mm256_div_ps(red, _mm256_set1_ps(blockSize * blockSize));
            green = _mm256_div_ps(green, _mm256_set1_ps(blockSize * blockSize));
            blue = _mm256_div_ps(blue, _mm256_set1_ps(blockSize * blockSize));

            int* currentRed = (int*)&red;
            int* currentGreen = (int*)&green;
            int* currentBlue = (int*)&blue;

            for (int x = i; x < min(i + blockSize, height); x++) {
                for (int y = j; y < min(j + blockSize, width); y+=8) {
                    for (int k = 0; k < 8; k++) {
                        if (y + k < min(j + blockSize, width)) {
                            int currentindex = (x * width + y) * channels + k;

                            mosaicImage[currentindex] = currentRed[k];
                            mosaicImage[currentindex + 1] = currentGreen[k];
                            mosaicImage[currentindex + 2] = currentBlue[k];
                        }
                    }
                }
            }
        }
    }

    return mosaicImage;
}

double checkTimeGauss(unsigned char* (*function)(unsigned char*, int, int, int, float),
                      unsigned char* image, int width, int height, int channels, float sigma) {
    auto begin = chrono::steady_clock::now();
    function(image, width, height, channels, sigma);
    auto end = chrono::steady_clock::now();

    return chrono::duration_cast<chrono::microseconds> (end - begin).count() / 1000000.0;
}

double checkTimeMosaic(unsigned char* (*function)(unsigned char*, int, int, int, int),
                      unsigned char* image, int width, int height, int channels, int blockSize) {
    auto begin = chrono::steady_clock::now();
    function(image, width, height, channels, blockSize);
    auto end = chrono::steady_clock::now();

    return chrono::duration_cast<chrono::microseconds> (end - begin).count() / 1000000.0;
}

void showTimeConsistentlyGaussImage(unsigned char* image, int width, int height, int channels, float sigma,
                                    int kernelSize=20) {
    double sum = 0.0;
    int attemptsNumber = 1;
    for (int attempt = 0; attempt < attemptsNumber; attempt++) {
        sum += checkTimeGauss(
                reinterpret_cast<unsigned char *(*)(unsigned char *, int, int, int, float)>(&gaussBlur),
                image, width, height, channels, sigma);
    }

    cout << "Image size: " << width << "x" << height << endl;
    cout << "Number of pixels: " << width * height << endl;
    cout << "Mean time: " << sum / attemptsNumber << endl;
    cout << endl;
}

void showTimeConsistentlyMosaicFilter(unsigned char* image, int width, int height, int channels, int blockSize) {
    double sum = 0.0;
    int attemptsNumber = 100;
    for (int attempt = 0; attempt < attemptsNumber; attempt++) {
        sum += checkTimeMosaic(
                reinterpret_cast<unsigned char *(*)(unsigned char *, int, int, int, int)>(&mosaicFilter),
                image, width, height, channels, blockSize);
    }

    cout << "Image size: " << width << "x" << height << endl;
    cout << "Number of pixels: " << width * height << endl;
    cout << "Mean time: " << sum / attemptsNumber << endl;
    cout << endl;
}

void showTimeOpenMPGaussImage(unsigned char* image, int width, int height, int channels, float sigma,
                                    int kernelSize=20) {
    double sum = 0.0;
    int attemptsNumber = 1;
    for (int attempt = 0; attempt < attemptsNumber; attempt++) {
        sum += checkTimeGauss(
                reinterpret_cast<unsigned char *(*)(unsigned char *, int, int, int, float)>(&openMPGaussBlur),
                image, width, height, channels, sigma);
    }

    cout << "Image size: " << width << "x" << height << endl;
    cout << "Number of pixels: " << width * height << endl;
    cout << "Mean time: " << sum / attemptsNumber << endl;
    cout << endl;
}

void showTimeOpenMPMosaicFilter(unsigned char* image, int width, int height, int channels, int blockSize) {
    double sum = 0.0;
    int attemptsNumber = 100;
    for (int attempt = 0; attempt < attemptsNumber; attempt++) {
        sum += checkTimeMosaic(
                reinterpret_cast<unsigned char *(*)(unsigned char *, int, int, int, int)>(&openMPMosaicFilter),
                image, width, height, channels, blockSize);
    }

    cout << "Image size: " << width << "x" << height << endl;
    cout << "Number of pixels: " << width * height << endl;
    cout << "Mean time: " << sum / attemptsNumber << endl;
    cout << endl;
}

void showTimeVectorizedGaussImage(unsigned char* image, int width, int height, int channels, float sigma,
                                  int kernelSize=20) {
    double sum = 0.0;
    int attemptsNumber = 1;
    for (int attempt = 0; attempt < attemptsNumber; attempt++) {
        sum += checkTimeGauss(
                reinterpret_cast<unsigned char *(*)(unsigned char *, int, int, int, float)>(&vectorizedGaussBlur),
                image, width, height, channels, sigma);
    }

    cout << "Image size: " << width << "x" << height << endl;
    cout << "Number of pixels: " << width * height << endl;
    cout << "Mean time: " << sum / attemptsNumber << endl;
    cout << endl;
}

void showTimeVectorizedMosaicFilter(unsigned char* image, int width, int height, int channels, int blockSize) {
    double consistentlySum = 0.0;
    int attemptsNumber = 100;
    for (int attempt = 0; attempt < attemptsNumber; attempt++) {
        consistentlySum += checkTimeMosaic(
                reinterpret_cast<unsigned char *(*)(unsigned char *, int, int, int, int)>(&vectorizedMosaicFilter),
                image, width, height, channels, blockSize);
    }

    cout << "Image size: " << width << "x" << height << endl;
    cout << "Number of pixels: " << width * height << endl;
    cout << "Mean time: " << consistentlySum / attemptsNumber << endl;
    cout << endl;
}

void showTimeConsistentlyGauss(Images images, float sigma=7.2) {
    cout << "CONSICTENTLY GAUSS BLUR" << endl;
    cout << endl;
    for (int i = 0; i < 6; i++) {
        showTimeConsistentlyGaussImage(images.imagesArray[i], images.sizes[i], images.sizes[i],
                                       images.channels[i], sigma);
    }
}

void showTimeConsistentlyMosaic(Images images, int blockSize) {
    cout << "CONSICTENTLY MOSAIC FILTER" << endl;
    cout << endl;
    for (int i = 0; i < 6; i++) {
        showTimeOpenMPMosaicFilter(images.imagesArray[i], images.sizes[i], images.sizes[i],
                                         images.channels[i], blockSize);
    }
}

void showTimeOpenMPGauss(Images images, float sigma=7.2) {
    cout << "OPEN MP GAUSS BLUR" << endl;
    cout << endl;
    for (int i = 0; i < 6; i++) {
        showTimeOpenMPGaussImage(images.imagesArray[i], images.sizes[i], images.sizes[i],
                                       images.channels[i], sigma);
    }
}

void showTimeOpenMPMosaic(Images images, int blockSize) {
    cout << "OPEN MP MOSAIC FILTER" << endl;
    cout << endl;
    for (int i = 0; i < 6; i++) {
        showTimeConsistentlyMosaicFilter(images.imagesArray[i], images.sizes[i], images.sizes[i],
                                 images.channels[i], blockSize);
    }
}

void showTimeVectorizedGauss(Images images, float sigma=7.2) {
    cout << "VECTORIZED GAUSS BLUR" << endl;
    cout << endl;
    for (int i = 0; i < 6; i++) {
        showTimeVectorizedGaussImage(images.imagesArray[i], images.sizes[i], images.sizes[i],
                                     images.channels[i], sigma);
    }
}

void showTimeVectorizedMosaic(Images images, int blockSize) {
    cout << "VECTORIZED MOSAIC FILTER" << endl;
    cout << endl;
    for (int i = 0; i < 6; i++) {
        showTimeVectorizedMosaicFilter(images.imagesArray[i], images.sizes[i], images.sizes[i],
                                     images.channels[i], blockSize);
    }
}

void calculateGaussBlur(Images images) {
    const char* imageName = "image.png";
    cout << "Received: " << imageName << endl;

    int width, height, channels;
    unsigned char* image = stbi_load(imageName, &width, &height, &channels, 0);
    cout << "width: " << width << "\n" << "height: " << height << "\n" << "channels: " << channels << endl;

    cout << endl;
    cout << "-------------------------------------------------" << endl;
    cout << "Gaussian Blur\n";
    cout << "-------------------------------------------------" << endl;
    cout << endl;

    float sigma;
    cout << "Enter sigma:" << endl;
    cin >> sigma;
    cout << endl;

    unsigned char* gaussImage = gaussBlur(image, width, height, channels, sigma);
    stbi_write_png("blured_image.png", width, height, channels, gaussImage, width*channels);
    cout << "Image saved as \"blured_image.png\"\n" << endl;

    cout << "Time analyse: " << endl;
    cout << "-------------------------------------------------" << endl;

    showTimeConsistentlyGauss(images);
    showTimeOpenMPGauss(images);
    showTimeVectorizedGauss(images);
}

void calculateMosaicFilter(Images images) {
    const char* imageName = "image.png";
    int width, height, channels;
    unsigned char* image = stbi_load(imageName, &width, &height, &channels, 0);

    cout << endl;
    cout << "-------------------------------------------------" << endl;
    cout << "Mosaic filter\n";
    cout << "-------------------------------------------------" << endl;
    cout << endl;

    int blockSize;
    cout << "Enter block size:" << endl;
    cin >> blockSize;
    cout << endl;

    unsigned char* mosaicImage = mosaicFilter(image, width, height, channels, blockSize);
    stbi_write_png("mosaic_image.png", width, height, channels, mosaicImage, width*channels);
    cout << "Image saved as \"mosaic_image.png\"\n" << endl;

    cout << "Time analyse: " << endl;
    cout << "-------------------------------------------------" << endl;

    showTimeConsistentlyMosaic(images, blockSize);
    showTimeOpenMPMosaic(images, blockSize);
    showTimeVectorizedMosaic(images, blockSize);
}

Images getImages() {
    Images images;
    for (int i = 0; i < 6; i++) {
        int w, h, ch;
        unsigned char* image = stbi_load(images.imagesNames[i], &w, &h, &ch, 0);

        images.imagesArray.push_back(image);
        images.sizes.push_back(w);
        images.channels.push_back(ch);
    }

    return images;
}

int main() {
    // get images with different sizes for time analyse
    Images images = getImages();

    calculateGaussBlur(images);
    calculateMosaicFilter(images);

    return 0;
}
