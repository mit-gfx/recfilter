#pragma warning(disable:4819)

/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Recursive Gaussian filter
    sgreen 8/1/08

    This code sample implements a Gaussian blur using Deriche's recursive method:
    http://citeseer.ist.psu.edu/deriche93recursively.html

    This is similar to the box filter sample in the SDK, but it uses the previous
    outputs of the filter as well as the previous inputs. This is also known as an
    IIR (infinite impulse response) filter, since its response to an input impulse
    can last forever.

    The main advantage of this method is that the execution time is independent of
    the filter width.

    The GPU processes columns of the image in parallel. To avoid uncoalesced reads
    for the row pass we transpose the image and then transpose it back again
    afterwards.

    The implementation is based on code from the CImg library:
    http://cimg.sourceforge.net/
    Thanks to David Tschumperl� and all the CImg contributors!
*/

// CUDA includes and interop headers
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>      // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX(a,b) ((a > b) ? a : b)

#define USE_SIMPLE_FILTER 0

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD  0.15f

float sigma = 10.0f;
int order   = 0;
int nthreads= 64;  // number of threads per block

unsigned int *h_img = NULL;
unsigned int *d_img = NULL;
unsigned int *d_temp = NULL;

StopWatchInterface *timer = 0;

extern "C"
void transpose(unsigned int *d_src, unsigned int *d_dest, unsigned int width, int height);

extern "C"
double gaussianFilterRGBA(uint *d_src, uint *d_dest, uint *d_temp, int width, int height, float sigma, int order, int nthreads, StopWatchInterface *timer);

void cleanup();


void benchmark(int image_width)
{
    unsigned int width = image_width;
    unsigned int height= image_width;

    // allocate memory for result
    unsigned int *d_result;
    unsigned int size = width * height * sizeof(unsigned int);

    h_img = (unsigned int*) malloc(width*height*sizeof(float));
    for (int i=0; i<width*height; i++) {
        h_img[i] = rand()/float(RAND_MAX);
    }

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_img, size));
    checkCudaErrors(cudaMalloc((void **) &d_temp, size));
    checkCudaErrors(cudaMalloc((void **) &d_result, size));
    checkCudaErrors(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));

    sdkCreateTimer(&timer);

    // warm-up
    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads, timer);

    checkCudaErrors(cudaDeviceSynchronize());

    // execute the kernel
    const int iCycles = 100;
    double dProcessingTime = 0.0;

    for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads, timer);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    // Get average computation time
    dProcessingTime /= (double)iCycles;

    // fprintf(stderr, "%d\t%f ms\n", width, dProcessingTime);
    float throughput = (width*width*1000.0f)/(dProcessingTime*1024*1024);
    fprintf(stderr, "%d\t%f\t%f\n", width, dProcessingTime, throughput);

    checkCudaErrors(cudaFree(d_result));
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Benchmark or AutoTest mode detected, no OpenGL
    findCudaDevice(argc, (const char **)argv);

    int inc_w = 64;
    int min_w = inc_w;
    int max_w = 4096;
    for (int w=min_w; w<=max_w; w+=inc_w) {
        benchmark(w);
    }

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
