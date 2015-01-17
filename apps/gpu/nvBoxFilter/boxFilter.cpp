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
    Image box filtering example

    This sample uses CUDA to perform a simple box filter on an image
    and uses OpenGL to display the results.

    It processes rows and columns of the image in parallel.

    The box filter is implemented such that it has a constant cost,
    regardless of the filter width.

    Press '=' to increment the filter radius, '-' to decrease it

    Version 1.1 - modified to process 8-bit RGBA images
*/

// CUDA utilities and system includes
#include <cuda_runtime.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY     10 //ms

#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10

int filter_radius   = 14;
int nthreads        = 64;
float *h_img        = NULL;
float *d_img        = NULL;
float *d_temp       = NULL;

StopWatchInterface *kernel_timer = NULL;

extern "C" void runBenchmark(int w, int iter);
extern "C" void initTexture(int width, int height, void *pImage, bool useRGBA);
extern "C" void freeTextures();

// These are CUDA functions to handle allocation and launching the kernels
extern "C" double boxFilter(float *d_src, float *d_temp, float *d_dest, int width, int height,
                            int radius, int iterations, int nthreads, StopWatchInterface *timer);

void cleanup()
{
    sdkDeleteTimer(&kernel_timer);

    if (h_img)
    {
        cudaFree(h_img);
        h_img=NULL;
    }

    if (d_img)
    {
        cudaFree(d_img);
        d_img=NULL;
    }

    if (d_temp)
    {
        cudaFree(d_temp);
        d_temp=NULL;
    }

    // Refer to boxFilter_kernel.cu for implementation
    freeTextures();
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple benchmark test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runBenchmark(int image_width, int iterations)
{
    unsigned int width = image_width;
    unsigned int height= image_width;

    h_img = (float*) malloc(width*height*sizeof(float));
    for (int i=0; i<width*height; i++) {
        h_img[i] = rand()/float(RAND_MAX);
    }

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_img, (width * height * sizeof(float))));
    checkCudaErrors(cudaMalloc((void **) &d_temp,(width * height * sizeof(float))));
    initTexture(width, height, h_img, false);
    sdkCreateTimer(&kernel_timer);

    // warm-up
    boxFilter(d_img, d_temp, d_temp, width, height, filter_radius, iterations, nthreads, kernel_timer);
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStartTimer(&kernel_timer);
    // Start round-trip timer and process iCycles loops on the GPU
    const int iCycles = 100;
    double dProcessingTime = 0.0;

    for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += boxFilter(d_img, d_temp, d_img, width, height, filter_radius, iterations, nthreads, kernel_timer);
    }

    // check if kernel execution generated an error and sync host
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&kernel_timer);

    // Get average computation time
    dProcessingTime /= (double)iCycles;

    // fprintf(stderr, "%d\t%f ms\n", width, dProcessingTime);
    float throughput = (width*width*1000.0f)/(dProcessingTime*1024*1024);
    fprintf(stderr, "%d\t%f\t%f\n", width, dProcessingTime, throughput);
}

bool checkCUDAProfile(int dev, int min_runtime, int min_compute)
{
    int runtimeVersion = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    fprintf(stdout,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    cudaRuntimeGetVersion(&runtimeVersion);
    fprintf(stdout,"  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    fprintf(stdout,"  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

    if (runtimeVersion >= min_runtime && ((deviceProp.major<<4) + deviceProp.minor) >= min_compute)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findCapableDevice(int argc, char **argv)
{
    int dev;
    int bestDev = -1;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        fprintf(stdout, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    if (deviceCount==0)
    {
        fprintf(stdout,"There are no CUDA capabile devices.\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        fprintf(stdout,"Found %d CUDA Capable device(s) supporting CUDA\n", deviceCount);
    }

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (checkCUDAProfile(dev, MIN_RUNTIME_VERSION, MIN_COMPUTE_VERSION))
        {
            fprintf(stdout,"\nFound CUDA Capable Device %d: \"%s\"\n", dev, deviceProp.name);

            if (bestDev == -1)
            {
                bestDev = dev;
                fprintf(stdout, "Setting active device to %d\n", bestDev);
            }
        }
    }

    if (bestDev == -1)
    {
        fprintf(stdout, "\nNo configuration with available capabilities was found.  Test has been waived.\n");
        fprintf(stdout, "The CUDA Sample minimum requirements:\n");
        fprintf(stdout, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION/16, MIN_COMPUTE_VERSION%16);
        fprintf(stdout, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION/1000, (MIN_RUNTIME_VERSION%100)/10);
        exit(EXIT_SUCCESS);
    }

    return bestDev;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    int iterations = 1;

    if (argc == 2) {
        iterations = atoi(argv[1]);
    } else {
        printf("Usage: nvBoxFilter [filter iterations]");
        return -1;
    }

    int devID = findCudaDevice(argc, (const char **)argv);

    int inc_w = 64;
    int min_w = inc_w;
    int max_w = 4096;
    for (int w=min_w; w<=max_w; w+=inc_w) {
        runBenchmark(w, iterations);
        cleanup();
    }

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    return 0;
}
