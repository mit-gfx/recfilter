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

// Includes
//*****************************************************************************

// Includes
#include <memory>
#include <iostream>
#include <cassert>

// utilities, system and OpenCL includes
#include <oclUtils.h>
#include <shrQATest.h>

#ifndef min
#define min(a,b) (a < b ? a : b);
#endif


// Defines and globals for box filter processing demo
//*****************************************************************************
#define REFRESH_DELAY	  10 //ms

cl_uint uiNumOutputPix = 64;                   // Default output pix per workgroup... may be modified depending HW/OpenCl caps
cl_uint iRadius = 10;                         // initial radius of 2D box filter mask
float fScale = 1.0f/(2.0f * iRadius + 1.0f);  // precalculated GV rescaling value
cl_int iRadiusAligned;

// Global declarations
//*****************************************************************************
// Image data vars
unsigned int uiImageWidth = 0;      // Image width
unsigned int uiImageHeight = 0;     // Image height
unsigned int* uiInput = NULL;       // Host buffer to hold input image data
unsigned int* uiOutput = NULL;      // Host buffer to hold output image data
unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data

// app configuration parms
const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
bool bFilter = true;                // state var for whether filter is enaged or not
int iProcFlag = 0;                  // 0 = GPU, 1 = CPU
int iTestSets = 3;                  // # of loop set retriggers before auto exit when bNoPrompt = shrTrue
int iCycles  = 100;                 // number of runs for profiling

// OpenCL vars
const char* clSourcefile = "BoxFilter.cl";  // OpenCL kernel source file
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation
cl_platform_id cpPlatform;          // OpenCL platform
cl_context cxGPUContext;            // OpenCL context
cl_command_queue cqCommandQueue;    // OpenCL command que
cl_device_id* cdDevices = NULL;     // device list
cl_uint uiNumDevsUsed = 1;          // Number of devices used in this sample
cl_program cpProgram=0;             // OpenCL program
cl_kernel ckBoxRowsLmem=0;          // OpenCL Kernel for row sum (using lmem)
cl_kernel ckBoxRowsTex=0;           // OpenCL Kernel for row sum (using 2d Image/texture)
cl_kernel ckBoxColumns=0;           // OpenCL for column sum and normalize
cl_mem cmDevBufIn=0;                // OpenCL device memory object (buffer or 2d Image) for input data
cl_mem cmDevBufTemp=0;              // OpenCL device memory temp buffer object
cl_mem cmDevBufOut=0;               // OpenCL device memory output buffer object
cl_mem cmCL_PBO=0;					// OpenCL representation of GL pixel buffer
cl_image_format InputFormat;        // OpenCL format descriptor for 2D image useage
cl_sampler RowSampler=0;            // Image Sampler for box filter processing with texture (2d Image)
size_t szBuffBytes;                 // Size of main image buffers
size_t szGlobalWorkSize[2];         // global # of work items
size_t szLocalWorkSize[2];          // work group # of work items
size_t szMaxWorkgroupSize = 512;    // initial max # of work items
size_t szParmDataBytes;			    // Byte size of context information
size_t szKernelLength;			    // Byte size of kernel code
cl_int ciErrNum;			        // Error code var


// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
double BoxFilterGPU(int numBoxFilterIterations, unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);
void ResetKernelArgs(unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);

// Helpers
void TestNoGL(int numBoxFilterIterations);
static inline size_t DivUp(size_t dividend, size_t divisor);
void Exit(int);
void Cleanup();
void (*pCleanup)(int) = &Exit;

// Helper to get next up value for integer division
//*****************************************************************************
static inline size_t DivUp(size_t dividend, size_t divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

// Main program
//*****************************************************************************
int main(int argc, char** argv)
{
    cl_uint uiNumDevices = 0;           // Number of devices available
    cl_uint uiTargetDevice = 0;	        // Default Device to compute on
    cl_uint uiNumComputeUnits;          // Number of compute units (SM's on NV GPU)

    int numBoxFilterIterations = 0;

    if (argc == 3) {
        numBoxFilterIterations = atoi(argv[1]);
        uiTargetDevice = atoi(argv[2]);
    } else {
        printf("Usage: oclBoxFilter [number of filtering iterations] [OpenCL device]");
        exit(EXIT_FAILURE);
    }

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    //Get all the devices
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set target device and Query number of compute units on uiTargetDevice
    printf("Devices Available = %u\n", uiNumDevices);
    for (int i=0; i<uiNumDevices; i++) {
        ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
        printf("Device %u: ", i);
        oclPrintDevName(LOGBOTH, cdDevices[i]);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        printf(" Compute Units = %u\n", uiNumComputeUnits);
    }
    printf("Using Device %u: \n", uiTargetDevice);

    //Create the context
    cxGPUContext = clCreateContext(0, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    int min_w = 16;
    int max_w = 4096;
    int inc_w = 32;

    for (int N=min_w; N<=max_w; N+=inc_w)
    {
        // Allocate OpenCL object for the source data
        // Buffer in device GMEM
        // create an input image
        // Allocate intermediate and output host image buffers
        uiImageWidth = N;
        uiImageHeight= N;
        szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
        uiInput = (unsigned int*)malloc(szBuffBytes);
        uiOutput= (unsigned int*)malloc(szBuffBytes);
        uiTemp  = (unsigned int*)malloc(szBuffBytes);

        srand(0);
        for (int x=0; x<uiImageWidth; x++) {
            for (int y=0; y<uiImageHeight; y++) {
                uiInput[uiImageWidth*y + x] = rand() % 255;
            }
        }

        cmDevBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Allocate the OpenCL intermediate and result buffer memory objects on the device GMEM
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
        std::string sBuildOpts = " -cl-fast-relaxed-math";
        sBuildOpts  += " -D USELMEM";

        // mac
#ifdef MAC
        sBuildOpts  += " -DMAC";
#endif
        //--------------------------------

        // Build the program
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, sBuildOpts.c_str(), NULL, NULL);
        if (ciErrNum != CL_SUCCESS)
        {
            // If build problem, write out standard ciErrNum, Build Log and PTX, then cleanup and exit
            oclLogBuildInfo(cpProgram, cdDevices[uiTargetDevice]);
            oclLogPtx(cpProgram, cdDevices[uiTargetDevice], "oclBoxFilter.ptx");
            shrQAFinish(argc, (const char **)argv, QA_FAILED);
            Exit(EXIT_FAILURE);
        }

        // Create kernels
        {
            ckBoxRowsLmem = clCreateKernel(cpProgram, "BoxRowsLmem", &ciErrNum);
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

            // get max work group size
            ciErrNum = clGetKernelWorkGroupInfo(ckBoxRowsLmem, cdDevices[uiTargetDevice],
                    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szMaxWorkgroupSize, NULL);
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        }
        ckBoxColumns = clCreateKernel(cpProgram, "BoxColumns", &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // set the kernel args
        ResetKernelArgs(uiImageWidth, uiImageHeight, iRadius, fScale);

        // init running timers
        shrDeltaT(0);   // timer 0 used for computation timing

        // test sequence
        TestNoGL(numBoxFilterIterations);

        // delete the image buffers
        Cleanup();
    }

    exit(EXIT_SUCCESS);
}

// Function to set kernel args that only change outside of GLUT loop
//*****************************************************************************
void ResetKernelArgs(unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    // Set the Argument values for the row kernel
    {
        // (lmem version)
        iRadiusAligned = ((r + 15)/16) * 16;
        if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + r))
        {
            uiNumOutputPix = (cl_uint)szMaxWorkgroupSize - iRadiusAligned - r;
        }
        ciErrNum = clSetKernelArg(ckBoxRowsLmem, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 2, (iRadiusAligned + uiNumOutputPix + r) * sizeof(cl_uchar4), NULL);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 3, sizeof(unsigned int), (void*)&uiWidth);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 4, sizeof(unsigned int), (void*)&uiHeight);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 5, sizeof(int), (void*)&r);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 6, sizeof(int), (void*)&iRadiusAligned);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 7, sizeof(float), (void*)&fScale);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 8, sizeof(unsigned int), (void*)&uiNumOutputPix);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        //        printf("clSetKernelArg (0-8) ckBoxRowsLmem...\n");
    }

    // Set the Argument values for the column kernel
    ciErrNum  = clSetKernelArg(ckBoxColumns, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 2, sizeof(unsigned int), (void*)&uiWidth);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 3, sizeof(unsigned int), (void*)&uiHeight);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 4, sizeof(int), (void*)&r);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 5, sizeof(float), (void*)&fScale);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    //    printf("clSetKernelArg (0-5) ckBoxColumns...\n\n");
}

// OpenCL computation function for GPU:
// Copies input data to the device, runs kernel, copies output data back to host
//*****************************************************************************
double BoxFilterGPU(int numBoxFilterIterations, unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    double dKernelTime = 0.0;

    // Sync host and start computation timer
    clFinish(cqCommandQueue);
    shrDeltaT(0);

    // perform box filter iterations
    for (int j=0; j<numBoxFilterIterations; j++) {
        // swap input and output buffers after the first iteration because the
        // second iterations needs to use the resulf of the first as input
        if (j > 0) {
            cl_mem c_temp = cmDevBufIn;
            cmDevBufIn = cmDevBufOut;
            cmDevBufOut= c_temp;
        }

        // Launch row kernel
        // lmem Version
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBoxRowsLmem, 2, NULL,
                szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Set global and local work sizes for column kernel
        szLocalWorkSize[0] = 64;
        szLocalWorkSize[1] = 1;
        szGlobalWorkSize[0] = szLocalWorkSize[0] * DivUp((size_t)uiWidth, szLocalWorkSize[0]);
        szGlobalWorkSize[1] = 1;

        // Launch column kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBoxColumns, 2, NULL,
                szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // sync host and return computation time
        clFinish(cqCommandQueue);

        // ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cmDevBufOut, CL_TRUE, 0, szBuffBytes, uiOutput, 0, NULL, NULL);
        // for (int x=0; x<uiImageWidth; x++) {
        //     for (int y=0; y<uiImageHeight; y++) {
        //         printf("%04d,%04d ", uiInput[uiImageWidth*y + x], uiOutput[uiImageWidth*y + x]);
        //     }
        //     printf("\n");
        // }
        // printf("--\n");
    }
    dKernelTime = shrDeltaT(0);

    //exit(0);

    return dKernelTime;
}

// Run a test sequence without any GL
//*****************************************************************************
void TestNoGL(int numBoxFilterIterations)
{
    // Copy input data from host to device
    {
        // lmem version
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevBufIn, CL_TRUE, 0, szBuffBytes, uiInput, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Set global and local work sizes for row kernel
        szLocalWorkSize[0] = (size_t)(iRadiusAligned + uiNumOutputPix + iRadius);   // Workgroup padded left and right
        szLocalWorkSize[1] = 1;
        szGlobalWorkSize[0] = szLocalWorkSize[0] * DivUp((size_t)uiImageWidth, (size_t)uiNumOutputPix);
        szGlobalWorkSize[1] = uiImageHeight;
    }
    clFinish(cqCommandQueue);

    // run once to warm up the opencl driver
    BoxFilterGPU (numBoxFilterIterations, uiImageWidth, uiImageHeight, iRadius, fScale);
    clFinish(cqCommandQueue);

    // Start round-trip timer and process iCycles loops on the GPU
    double dProcessingTime = 0.0;
    for (int i = 0; i < iCycles; i++) {
        dProcessingTime += BoxFilterGPU(numBoxFilterIterations, uiImageWidth, uiImageHeight, iRadius, fScale);
    }
    clFinish(cqCommandQueue);

    // Get round-trip and average computation time
    dProcessingTime /= (double)iCycles;

    fprintf(stderr, "%u\t%f\n", uiImageWidth, dProcessingTime*1000);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup()
{
    if(cSourceCL)     { free(cSourceCL);                   cSourceCL=0;      }
    if(uiInput)       { free(uiInput);                     uiInput=0;        }
    if(uiOutput)      { free(uiOutput);                    uiOutput=0;       }
    if(uiTemp)        { free(uiTemp);                      uiTemp=0;         }
    if(ckBoxColumns)  { clReleaseKernel(ckBoxColumns);     ckBoxColumns=0;   }
    if(ckBoxRowsTex)  { clReleaseKernel(ckBoxRowsTex);     ckBoxRowsTex=0;   }
    if(ckBoxRowsLmem) { clReleaseKernel(ckBoxRowsLmem);    ckBoxRowsLmem=0;  }
    if(cpProgram)     { clReleaseProgram(cpProgram);       cpProgram=0;      }
    if(RowSampler)    { clReleaseSampler(RowSampler);      RowSampler=0;     }
    if(cmDevBufIn)    { clReleaseMemObject(cmDevBufIn);    cmDevBufIn=0;     }
    if(cmDevBufTemp)  { clReleaseMemObject(cmDevBufTemp);  cmDevBufTemp=0;   }
    if(cmDevBufOut)   { clReleaseMemObject(cmDevBufOut);   cmDevBufOut=0;    }
    if(cmCL_PBO)      { clReleaseMemObject(cmCL_PBO);      cmCL_PBO=0;       }
}

void Exit(int exitcode)
{
    // Cleanup allocated objects
    Cleanup();
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cdDevices)free(cdDevices);
    exit(exitcode);
}
