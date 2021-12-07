/*  David Cox
    GPU CSCD445
    Final Project
    Copyright 2021
*/
/*
    This program uses an input file of integers each separated by a new line as it's input.
*/
/*
    Limit to the program is that the GPU can not have an input greater than 134,215,680
    This limitation is because the limit on blocks is 65,536 with 1024 threads per block
    gives 67,108,864 threads total.  Each thread handles 2 elements which gives you 134,215,680.

    To increase this limit a new dimesion of blocks will need to be added.  Currently Cuda supports
    upto 65535 blocks in each dimension up to 3 dimensions.  This means if coded correctly, this could 
    theoretically handle 65535*65535*65535 = 281,462,092,005,375 blocks, 288,217,182,213,504,000 threads, 
    or 576,434,364,427,008,000 input size!
*/
/*
    Please check online to see which cuda architecture you should be using.  This is built with -arch=sm_37 
    but that will be depricated soon.  The make file should be updated to the current arch that your GPU 
    supports and the program should run the same.
*/
/*
    To use this program, use the attached make file and in the command line type make.  This will compile the
    code in the current directory.  If you get any errors, please double check your architecture and that you
    have nvcc for compiling CUDA code.  Once compiled using the attached make the program name will be project.

    Launch the program by using the program name with upto 3 additional arguments.  The first argument must be
    your input file name. The next two inputs are optional and your options are: -c, --cpu, -d, --descending,
    -a, --ascending.  The default with no options is that the input will be run through both GPU sorting types
    with their results displayed on the screen followed by the data output in ascending order.
*/

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>

#include "files.h"
#include "mergeCPU.h"
#include "mergeGPU.h"

int* createGPUArray(int* array, int numberElements, int paddedElements, size_t arrayBytes);
int paddingGPU(int numberElements);
void usage();
int validOption(const char* option);
void printArray(int array[], int size);
void outputArray(int array[], int size, bool ascending, char fileName[]);
int determineThreadsPerBlock(int inputSize);
bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);
void getDeviceInfo();
double currentTime();

int main(int argc, const char *argv[])
{
    char cpuOutputName[] = "CPUoutput.txt";
    char gpuNaiveOutputName[] = "GPUNaiveoutput.txt";
    char gpuBinaryOutputName[] = "GPUBinaryoutput.txt";
    int maxArgumentSize = 15;
    bool printAscending = true;
    bool runCPU = false;
    int numberElements;
    char option1[maxArgumentSize];
    char option2[maxArgumentSize];

    if (argc > 4 || argc < 2)
    {
        usage();
        return -1;
    }//end if not correct argument count
    else if (argc == 2)
    {
        option1[0] = '\0';
        option2[0] = '\0';
    }//end if no options selected    
    else if (argc == 3)
    {
        if(!validOption(argv[2]))
        {
            printf("Option 1 invalid\n");
            return -1;
        }//end if option 1 is invalid

        strcpy(option1, argv[2]);
        option2[0] = '\0';
    }//end if 1 option chosen
    else
    {
        if(!validOption(argv[2]) || !validOption(argv[3]))
        {
            printf("One option is invalid\n");
            return -1;
        }//end if option 1 or 2 is invalid

        strcpy(option1, argv[2]);
        strcpy(option2, argv[3]);
        if (strcasecmp(option1, "-d") == 0 && strcasecmp(option2, "-a") == 0 || strcasecmp(option1, "-a") == 0 && strcasecmp(option2, "-d") == 0 ||
            strcasecmp(option1, "--descending") == 0 && strcasecmp(option2, "--ascending") == 0 || strcasecmp(option1, "--ascending") == 0 && strcasecmp(option2, "--descending") == 0)
            {
                printf("BOTH sorting options chosen\n");
                usage();
                return -1;
            }//end if both ascending and descending chosen

    }//end if 2 options chosen

    if (option1[0] != '\0' && (strcasecmp(option1, "-d") == 0 || strcasecmp(option1, "--descending") == 0) || option2[0] != '\0' && (strcasecmp(option2, "-d") == 0 || strcasecmp(option2, "--descending") == 0))
    {
        printAscending = false;
    }//end if descending was chosen

    if (option1[0] != '\0' && (strcasecmp(option1, "-c") == 0 || strcasecmp(option1, "--cpu") == 0) || option2[0] != '\0' && (strcasecmp(option2, "-c") == 0 || strcasecmp(option2, "--cpu") == 0))
    {
        runCPU = true;
    }//end if CPU was chosen

    numberElements = countFileLines(argv[1]);
    size_t arrayBytes = numberElements * sizeof(int);
    int* inputArray = (int*) malloc(arrayBytes);
    retrieveFileArray(argv[1], inputArray);
    double startTime, finishTime, cpuTime, gpuTimeNaive, gpuTimeBinary;
    int paddedElements, threadsPerBlock, tileSize, numberBlocks, sharedMemorySize;
    size_t gpuArrayBytes;
    int *h_array, *d_arrayIn, *d_arrayOut, *temp;

    printf("\n");
    printf("Number of elements: %d\n\n", numberElements);

 //Set up GPU Naive
    printf("\n");
    printf("Starting NAIVE GPU\n");
    paddedElements = paddingGPU(numberElements);
    threadsPerBlock = determineThreadsPerBlock(paddedElements);

    //each thread handles 2 elements, make tileSize accordingly
    tileSize = threadsPerBlock * 2;

    //shared memory will have room for each element twice, input and output.
    //shared memory will have to twice the tilesize
    sharedMemorySize = threadsPerBlock * 2 * sizeof(int) * 2;    
    
    numberBlocks = (paddedElements / 2) / threadsPerBlock;
    gpuArrayBytes = paddedElements * sizeof(int);
   
    printf("Number of padded elements: %d\n", paddedElements);
    printf("Number of threads per block:%d\n", threadsPerBlock);
    printf("Number of blocks:%d\n", numberBlocks);
    printf("\n");

    h_array = createGPUArray(inputArray, numberElements, paddedElements, gpuArrayBytes);

    cudaMalloc((void**) &d_arrayIn, gpuArrayBytes);
    cudaMalloc((void**) &d_arrayOut, gpuArrayBytes);

    startTime = currentTime();
    cudaMemset(d_arrayOut, 0 ,gpuArrayBytes);
    cudaMemcpy(d_arrayIn, h_array, gpuArrayBytes, cudaMemcpyHostToDevice);

    mergeSortNaiveStep1<<<numberBlocks, threadsPerBlock, sharedMemorySize>>>(d_arrayIn, paddedElements, tileSize);
    cudaDeviceSynchronize();
    
    //step two of naive merge sort.
    for (tileSize *= 2; tileSize <= paddedElements; tileSize *= 2)
    {
        cudaDeviceSynchronize();
        numberBlocks /= 2;
        mergeSortNaiveStep2<<<numberBlocks, threadsPerBlock>>>(d_arrayIn, d_arrayOut, paddedElements, tileSize);

        temp = d_arrayIn;
        d_arrayIn = d_arrayOut;
        d_arrayOut = temp;
    }//end for merge step 2

    cudaMemcpy(h_array, d_arrayIn, gpuArrayBytes, cudaMemcpyDeviceToHost);
    finishTime = currentTime();
    gpuTimeNaive = finishTime - startTime;

    outputArray(h_array, numberElements, printAscending, gpuNaiveOutputName);

    cudaFree(d_arrayIn);
    cudaFree(d_arrayOut);
    free(h_array);


 //Start GPU BINARY

    printf("\n");
    printf("Starting BINARY GPU\n");
    paddedElements = paddingGPU(numberElements);
    threadsPerBlock = determineThreadsPerBlock(paddedElements);

    //each thread handles 2 elements, make tileSize accordingly
    tileSize = 1;

    //shared memory will have room for each element twice, input and output.
    sharedMemorySize = threadsPerBlock * 2 * sizeof(int) * 2;    
    
    numberBlocks = (paddedElements / 2) / threadsPerBlock;
    gpuArrayBytes = paddedElements * sizeof(int);
    
    printf("Number of padded elements: %d\n", paddedElements);
    printf("Number of threads per block:%d\n", threadsPerBlock);
    printf("Initial tileSize:%d\n", tileSize);
    printf("Number of blocks:%d\n", numberBlocks);

    h_array = createGPUArray(inputArray, numberElements, paddedElements, gpuArrayBytes);

    cudaMalloc((void**) &d_arrayIn, gpuArrayBytes);
    cudaMalloc((void**) &d_arrayOut, gpuArrayBytes);

    startTime = currentTime();

    cudaMemset(d_arrayOut, 0 ,gpuArrayBytes);
    cudaMemcpy(d_arrayIn, h_array, gpuArrayBytes, cudaMemcpyHostToDevice);
    
    for (tileSize = 1; tileSize < paddedElements; tileSize *= 2)
    {
        mergeSortBinary<<<numberBlocks, threadsPerBlock>>>(d_arrayIn, d_arrayOut, tileSize);
        cudaDeviceSynchronize();
    
        temp = d_arrayIn;
        d_arrayIn = d_arrayOut;
        d_arrayOut = temp;
    }//end for merge step 2

    cudaMemcpy(h_array, d_arrayIn, gpuArrayBytes, cudaMemcpyDeviceToHost);

    finishTime = currentTime();
    gpuTimeBinary = finishTime - startTime;

    outputArray(h_array, numberElements, printAscending, gpuBinaryOutputName);

 //end GPU BINARY

    printf("\n");
    printf("GPU-Naive runtime: %f\n", gpuTimeNaive);
    printf("GPU-Binary runtime: %f\n", gpuTimeBinary);
    printf("GPU Speed up factor: %f\n", gpuTimeNaive / gpuTimeBinary);
    printf("\n");

    if (runCPU)
    {
        startTime = currentTime();
        mergeSortCPU(inputArray, 0, numberElements - 1);
        finishTime = currentTime();
        cpuTime = finishTime - startTime;

        printf("CPU runtime: %f\n", cpuTime);
        printf("Speed up time with Naive: %f\n", cpuTime - gpuTimeNaive);
        printf("Speed up time with Binary: %f\n", cpuTime - gpuTimeBinary);
        printf("CPU/Naive Speed up factor: %f\n", cpuTime/gpuTimeNaive);
        printf("CPU/Binary Speed up factor: %f\n", cpuTime/gpuTimeBinary);

        outputArray(inputArray, numberElements, printAscending, cpuOutputName);
    }//end if cpu merge runs
        
    free(h_array);
    free(inputArray);
    cudaFree(d_arrayIn);
    cudaFree(d_arrayOut);
}//end main()

int* createGPUArray(int* array, int numberElements, int paddedElements, size_t arrayBytes)
{
    int* GPUarray = (int*) malloc(arrayBytes);
    for (int i = 0; i < numberElements; i++)
    {
        GPUarray[i] = array[i];
    }//end coppy elements

    for (int i = numberElements; i < paddedElements; i ++)
    {
        GPUarray[i] = INT_MAX;
    }//end padding

    return GPUarray;
}//end createGPUArray()

int paddingGPU(int numberElements)
{
    if (isPowerOfTwo(numberElements))
    {
        return numberElements;
    }//end if elements are a power of two.

    else
    {
        return nextPowerOfTwo(numberElements);
    }//end else get power of two.

}//end paddingGPU()

int validOption(const char* option)
{
    //valid options are -c, -d, -a, --cpu, --descending, --ascending
    if (option[0] != '-')
    {
        printf("not a valid option, returning 0 (false)\n");
        return 0;
    }//end if option doesn't start with option mark
    else if (option[0] == '-' && option[1] == '-')
    {
        if (strcasecmp(option, "--descending") || strcasecmp(option, "--ascending") || strcasecmp(option, "--cpu"))
        {
            return 1;
        }//end if option matches proper word option

    }//end if option must be a word option with two option marks
    else if (option[0] == '-' && option[1] != '-')
    {
        if (option[1] == 'd' || option[1] == 'a' || option[1] == 'c')
        {
            return 1;
        }//end if option matches proper letter option

    }//end if option must be a letter option with one option mark

    return 0;
}//end validOption

void usage()
{
    printf("here is how you properly use this program\n");
    printf("usage: ./programName fileName option1 option2\n");
    printf("Options include: -c, --cpu to run on cpu as well\n");
    printf("                 -a, --ascending to write the list in ascending order (Default)\n");
    printf("                 -d, --descending to write the list in descending order\n");
    printf("only two options can be chosen at max, and only one sorting option can be chosen.\n");
}//end usage()

void printArray(int array[], int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d\n", array[i]);
    }//end for

}//end printArray()

void outputArray(int array[], int size, bool ascending, char fileName[])
{
    FILE * outputFile = fopen(fileName, "w");
    if (ascending)
    {
        for (int i = 0; i < size; i++)
        {
            fprintf(outputFile, "%d\n", array[i]);
        }//end for

    }//end if printing ascending
    else
    {
        for (int i = size - 1; i >= 0; i--)
        {
            fprintf(outputFile, "%d\n", array[i]);
        }//end for

    }//end if printing descending
}//end outputArray()

int determineThreadsPerBlock(int inputSize)
{

    int maxThreadCount = 512;
    int currentInputSize = inputSize;
    if (!isPowerOfTwo(inputSize))
    {
        currentInputSize = nextPowerOfTwo(inputSize);
    }//end if input is not power of two

    if (currentInputSize >= (maxThreadCount * 2))
    {
        return maxThreadCount;
    }//end if inputSize is large enough for max threads

    else
    {
        return currentInputSize / 2;
    }//end else currentInputSize is the new threads per block.

}//end determinThreadsPerBlock()

bool isPowerOfTwo(int x)
{
    return (ceil(log2((double)x)) == floor(log2((double)x)));
}//end isPowerOfTwo()

int nextPowerOfTwo(int x)
{
    return (pow(2, ceil(log((double)x)/log(2))));
}//end nextPowerOfTwo()

double currentTime()
{
   struct timeval now;
   gettimeofday(&now, NULL);
   
   return now.tv_sec + now.tv_usec/1000000.0;
}//end currentTime()

void getDeviceInfo()
{
    // Get CUDA Device Properties
    printf("Device Specs...\n");
    int num_cuda_devices;
    cudaGetDeviceCount(&num_cuda_devices);
    for(int i = 0; i < num_cuda_devices; i++) 
    {
        printf("Device %d - \n", i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("\tDevice Name: %s\n", prop.name);
        printf("\tMax Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("\tMax Threads Dim: %d\n", *prop.maxThreadsDim);
        printf("\tMax Grid Size: %d\n\n", *prop.maxGridSize);
    }
    printf("\n");
    printf("\n");
    printf("\n");
}//end getDeviceInfo