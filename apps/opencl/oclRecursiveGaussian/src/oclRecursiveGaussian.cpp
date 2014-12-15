/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Standard utilities and system includes, plus project specific items
//*****************************************************************************

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <cassert>

// Project Includes
#include "oclRecursiveGaussian.h"

// Shared QA Test Includes
#include <shrQATest.h>

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

// Defines and globals for recursive gaussian processing demo
//*****************************************************************************
#define REFRESH_DELAY	  10 //ms

float fSigma = 10.0f;               // filter sigma (blur factor)
int iOrder = 0;                     // filter order
int iTransposeBlockDim = 16;        // initial height and width dimension of 2D transpose workgroup
int iNumThreads = 64;	            // number of threads per block for Gaussian

// Image data vars
const char* cImageFile = "StoneRGB.ppm";
unsigned int uiImageWidth = 0;      // Image width
unsigned int uiImageHeight = 0;     // Image height
unsigned int* uiInput = NULL;       // Host buffer to hold input image data
unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data
unsigned int* uiOutput = NULL;      // Host buffer to hold output image data

// app configuration parms
const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
bool bFilter = true;                // state var for whether filter is enaged or not
int iProcFlag = 0;                  // 0 = GPU, 1 = CPU
int iTestSets = 3;                  // # of loop set retriggers before auto exit when bNoPrompt = shrTrue
int iCycles   = 100;

// OpenCL vars
const char* clSourcefile = "RecursiveGaussian.cl";  // OpenCL kernel source file
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL;                    // Buffer to hold source for compilation
cl_platform_id cpPlatform;          // OpenCL platform
cl_context cxGPUContext;            // OpenCL context
cl_command_queue cqCommandQueue;    // OpenCL command que
cl_device_id* cdDevices = NULL;     // device list
cl_uint uiNumDevsUsed = 1;          // Number of devices used in this sample
cl_program cpProgram=0;             // OpenCL program
cl_kernel ckRecursiveGaussianRGBA=0;// OpenCL Kernel for gaussian recursion
cl_kernel ckTranspose;              // OpenCL for transpose
cl_mem cmDevBufIn=0;                // OpenCL device memory input buffer object
cl_mem cmDevBufTemp=0;              // OpenCL device memory temp buffer object
cl_mem cmDevBufOut=0;               // OpenCL device memory output buffer object
size_t szBuffBytes;                 // Size of main image buffers
size_t szGaussGlobalWork;           // global # of work items in single dimensional range
size_t szGaussLocalWork;            // work group # of work items in single dimensional range
size_t szTransposeGlobalWork[2];    // global # of work items in 2 dimensional range
size_t szTransposeLocalWork[2];     // work group # of work items in a 2 dimensional range
size_t szParmDataBytes;			    // Byte size of context information
size_t szKernelLength;			    // Byte size of kernel code
cl_int ciErrNum;		            // Error code var
const char* cExecutableName;

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
double GPUGaussianFilterRGBA(GaussParms* pGP);
void GPUGaussianSetCommonArgs(GaussParms* pGP);
void TestNoGL(int iCycles);

// Helpers
void Cleanup();
void Exit(int iExitCode);
void (*pCleanup)(int) = &Exit;

// Main program
//*****************************************************************************
int main(int argc, char** argv)
{
    if (argc == 2) {
        iCycles = atoi(argv[1]);
    } else {
        printf("Usage: oclRecursiveGaussian [number of runs]");
        exit(EXIT_FAILURE);
    }

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    //Get all the devices
    cl_uint uiNumDevices = 0;           // Number of devices available
    cl_uint uiTargetDevice = 0;	        // Default Device to compute on
    cl_uint uiNumComputeUnits;          // Number of compute units (SM's on NV GPU)
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set target device and Query number of compute units on uiTargetDevice
    printf("  # of Devices Available = %u\n", uiNumDevices);
    uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    printf("  Using Device %u: ", uiTargetDevice);
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);
    ciErrNum = clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    printf("\n  # of Compute Units = %u\n\n", uiNumComputeUnits);

    //Create the context
    cxGPUContext = clCreateContext(0, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    for (int N=64; N<=8192; N+=64)
    {
        // Find the path from the exe to the image file and load the image
//        cPathAndName = shrFindFilePath(cImageFile, argv[0]);
//        oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
//        ciErrNum = shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
//        oclCheckErrorEX(ciErrNum, shrTRUE, pCleanup);

        uiImageWidth = N;
        uiImageHeight= N;

        // Allocate intermediate and output host image buffers
        szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
        uiTemp   = (unsigned int*) malloc(szBuffBytes);
        uiOutput = (unsigned int*) malloc(szBuffBytes);
        uiInput  = (unsigned int*) malloc(szBuffBytes);
        for (int x=0; x<uiImageWidth; x++) {
            for (int y=0; y<uiImageWidth; y++) {
                char r = rand() % 255;
                char g = rand() % 255;
                char b = rand() % 255;
                char a = 255;
                int pixel = 0;
                pixel |= a;
                pixel |= b<<8;
                pixel |= g<<16;
                pixel |= r<<24;
                uiInput[uiImageWidth*y+x] = pixel;
            }
       }

        // Allocate the OpenCL source, intermediate and result buffer memory objects on the device GMEM
        cmDevBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, szBuffBytes, NULL, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        cmDevBufTemp = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        cmDevBufOut = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, szBuffBytes, NULL, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Read the OpenCL kernel source in from file
        free(cPathAndName);
        cPathAndName = shrFindFilePath(clSourcefile, argv[0]);
        oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
        cSourceCL = oclLoadProgSource(cPathAndName, "// My comment\n", &szKernelLength);
        oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

        // Create the program
        cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Setup build options string
        //--------------------------------
        // Add mad option
        std::string sBuildOpts = " -cl-fast-relaxed-math";

        // Clamp to edge option
#ifdef CLAMP_TO_EDGE
        sBuildOpts  += " -D CLAMP_TO_EDGE";
#endif

        // mac
#ifdef MAC
        sBuildOpts  += " -DMAC";
#endif

        // Build the program
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, sBuildOpts.c_str(), NULL, NULL);
        if (ciErrNum != CL_SUCCESS)
        {
            // If build problem, write out standard ciErrNum, Build Log and PTX, then cleanup and exit
            oclLogBuildInfo(cpProgram, cdDevices[uiTargetDevice]);
            oclLogPtx(cpProgram, cdDevices[uiTargetDevice], "oclRecursiveGaussian.ptx");
            shrQAFinish(argc, (const char **)argv, QA_FAILED);
            Exit(EXIT_FAILURE);
        }

        // Create kernels
        ckRecursiveGaussianRGBA = clCreateKernel(cpProgram, "RecursiveGaussianRGBA", &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        ckTranspose = clCreateKernel(cpProgram, "Transpose", &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // check/reset work group size
        size_t wgSize;
        ciErrNum = clGetKernelWorkGroupInfo(ckTranspose, cdDevices[uiTargetDevice], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL);
        if (wgSize == 64)
        {
            iTransposeBlockDim = 8;
        }

        // Set unchanging local work sizes for gaussian kernels and transpose kernel
        szGaussLocalWork = iNumThreads;
        szTransposeLocalWork[0] = iTransposeBlockDim;
        szTransposeLocalWork[1] = iTransposeBlockDim;

        // init filter coefficients
        PreProcessGaussParms (fSigma, iOrder, &oclGP);

        // set common kernel args
        GPUGaussianSetCommonArgs (&oclGP);

        // init running timers
        shrDeltaT(0);   // timer 0 used for computation timing

        // Start main GLUT rendering loop for processing and rendering,
        // or otherwise run No-GL Q/A test sequence
        TestNoGL(iCycles);

        Cleanup();
    }

    Exit(EXIT_SUCCESS);
}

void TestNoGL(int iCycles)
{
    // Warmup call to assure OpenCL driver is awake
    GPUGaussianFilterRGBA(&oclGP);
    clFinish(cqCommandQueue);

    // Start round-trip timer and process iCycles loops on the GPU
    double dProcessingTime = 0.0;
    for (int i = 0; i < iCycles; i++) {
        dProcessingTime += GPUGaussianFilterRGBA(&oclGP);
    }
    // Get round-trip and average computation time
    dProcessingTime /= (double)iCycles;

    fprintf(stderr, "%u\t%f\n", uiImageWidth, dProcessingTime*1000);
}

// Function to set common kernel args that only change outside of GLUT loop
//*****************************************************************************
void GPUGaussianSetCommonArgs(GaussParms* pGP)
{
    // common Gaussian args
    // Set the Common Argument values for the Gaussian kernel
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 4, sizeof(float), (void*)&pGP->a0);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 5, sizeof(float), (void*)&pGP->a1);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 6, sizeof(float), (void*)&pGP->a2);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 7, sizeof(float), (void*)&pGP->a3);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 8, sizeof(float), (void*)&pGP->b1);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 9, sizeof(float), (void*)&pGP->b2);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 10, sizeof(float), (void*)&pGP->coefp);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 11, sizeof(float), (void*)&pGP->coefn);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set common transpose Argument values
    ciErrNum |= clSetKernelArg(ckTranspose, 4, sizeof(unsigned int) * iTransposeBlockDim * (iTransposeBlockDim + 1), NULL );
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

// 8-bit RGBA Gaussian filter for GPU on a 2D image using OpenCL
//*****************************************************************************
double GPUGaussianFilterRGBA(GaussParms* pGP)
{
    // var for kernel timing
    double dKernelTime = 0.0;

    // Copy input data from host to device
    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevBufIn, CL_TRUE, 0, szBuffBytes, uiInput, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // sync host and start timer
    clFinish(cqCommandQueue);
    shrDeltaT(0);

    // Set Gaussian global work dimensions, then set variable args and process in 1st dimension
    szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageWidth);

    // Set full Gaussian kernel variable arg values
    ciErrNum = clSetKernelArg(ckRecursiveGaussianRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 2, sizeof(unsigned int), (void*)&uiImageWidth);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 3, sizeof(unsigned int), (void*)&uiImageHeight);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch full Gaussian kernel on the data in one dimension
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckRecursiveGaussianRGBA, 1, NULL, &szGaussGlobalWork, &szGaussLocalWork, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set transpose global work dimensions and variable args
    szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageWidth);
    szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageHeight);
    ciErrNum = clSetKernelArg(ckTranspose, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckTranspose, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= clSetKernelArg(ckTranspose, 2, sizeof(unsigned int), (void*)&uiImageWidth);
    ciErrNum |= clSetKernelArg(ckTranspose, 3, sizeof(unsigned int), (void*)&uiImageHeight);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch transpose kernel in 1st direction
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckTranspose, 2, NULL, szTransposeGlobalWork, szTransposeLocalWork, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Reset Gaussian global work dimensions and variable args, then process in 2nd dimension
    // note width and height parameters flipped due to transpose
    szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageHeight);

    // Set full Gaussian kernel arg values
    ciErrNum = clSetKernelArg(ckRecursiveGaussianRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 2, sizeof(unsigned int), (void*)&uiImageHeight);
    ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 3, sizeof(unsigned int), (void*)&uiImageWidth);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch full Gaussian kernel on the data in the other dimension
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckRecursiveGaussianRGBA, 1, NULL, &szGaussGlobalWork, &szGaussLocalWork, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Reset transpose global work dimensions and variable args
    // note width and height parameters flipped due to 1st transpose
    szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageHeight);
    szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageWidth);
    ciErrNum = clSetKernelArg(ckTranspose, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckTranspose, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= clSetKernelArg(ckTranspose, 2, sizeof(unsigned int), (void*)&uiImageHeight);
    ciErrNum |= clSetKernelArg(ckTranspose, 3, sizeof(unsigned int), (void*)&uiImageWidth);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch transpose kernel in 2nd direction
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckTranspose, 2, NULL, szTransposeGlobalWork, szTransposeLocalWork, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // sync host and stop timer
    clFinish(cqCommandQueue);
    dKernelTime = shrDeltaT(0);

    // Copy results back to host, block until complete
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cmDevBufOut, CL_TRUE, 0, szBuffBytes, uiOutput, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

//    ciErrNum = shrSavePPM4ub("result.ppm", (unsigned char *)uiOutput, uiImageWidth, uiImageHeight);
//    oclCheckErrorEX(ciErrNum, shrTRUE, pCleanup);

    return dKernelTime;
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup()
{
    if(cSourceCL)     { free(cSourceCL);                   cSourceCL=0;      }
    if(uiInput)       { free(uiInput);                     uiInput=0;        }
    if(uiOutput)      { free(uiOutput);                    uiOutput=0;       }
    if(uiTemp)        { free(uiTemp);                      uiTemp=0;         }
    if(cpProgram)     { clReleaseProgram(cpProgram);       cpProgram=0;      }
    if(cmDevBufIn)    { clReleaseMemObject(cmDevBufIn);    cmDevBufIn=0;     }
    if(cmDevBufTemp)  { clReleaseMemObject(cmDevBufTemp);  cmDevBufTemp=0;   }
    if(cmDevBufOut)   { clReleaseMemObject(cmDevBufOut);   cmDevBufOut=0;    }
    if(cPathAndName)  { free(cPathAndName);                cPathAndName=0;   }
    if(ckTranspose)   { clReleaseKernel(ckTranspose);      ckTranspose=0;    }
    if(ckRecursiveGaussianRGBA){ clReleaseKernel(ckRecursiveGaussianRGBA); ckRecursiveGaussianRGBA=0;}
}
void Exit(int iExitCode)
{
    Cleanup();
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cdDevices)free(cdDevices);
    exit(iExitCode);
}
