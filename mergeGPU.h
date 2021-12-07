/*  Cody Bafus
    David Cox
    Colton Cronquist
    Anthony Hirt
    Jordan Meredith
    GPU CSCD445
    Final Project
*/

#ifndef HWF_MERGEGPU
#define HWF_MERGEGPU

__global__ void mergeSortNaiveStep1(int* inputArray, int dataElements, int tileSize);
__global__ void mergeSortNaiveStep2(int* inputArray, int* outputArray, int dataElements, int tileSize);
__device__ void mergeData(int* inputArray, int* outputArray, int left, int right, int elementCount, int startingOutput);
__global__ void mergeSortBinary(int* inputArray, int* outputArray, int tileSize);
__device__ int rank(int value, int* array, int tileSize, int upperSection);

#endif