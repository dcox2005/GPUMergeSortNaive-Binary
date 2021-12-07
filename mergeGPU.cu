/*  David Cox
    GPU CSCD445
    Final Project
*/

#include <stdio.h>
#include <stdlib.h>
#include "mergeGPU.h"

__global__ void mergeSortNaiveStep1(int* inputArray, int dataElements, int tileSize)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    int leftIndex, rightIndex;
    
    int globalInputID = globalID * 2;
    int secondGlobalID = globalInputID + 1;

    extern __shared__ int sharedData[];
    int threadInputID = threadIdx.x * 2;
    int secondThreadInputID = threadInputID + 1;
    sharedData[threadInputID] = inputArray[globalInputID];
    sharedData[secondThreadInputID] = inputArray[secondGlobalID];
    int outputStartingIndex = threadInputID + tileSize;

    leftIndex = threadInputID;
    __syncthreads();
   
    for (int step = 1; step < tileSize; step *= 2)
    {
        rightIndex = leftIndex + step;
        if (threadIdx.x % step == 0)
        {
            mergeData(sharedData, sharedData, leftIndex, rightIndex, step, outputStartingIndex);
        }//if thread still active

        __syncthreads();

        sharedData[threadInputID] = sharedData[threadInputID + tileSize];
        sharedData[secondThreadInputID] = sharedData[secondThreadInputID + tileSize];
        
        __syncthreads();
    }//end for merging
    
    inputArray[globalInputID] = sharedData[threadInputID];
    inputArray[secondGlobalID]= sharedData[secondThreadInputID];

}//end mergeSortNaiveStep1()

__global__ void mergeSortNaiveStep2(int* inputArray, int* outputArray, int dataElements, int tileSize)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalID >= (dataElements / tileSize))
    {
        return;
    }//end if thread not needed

    int leftIndex = globalID * tileSize; 
    int elementsCount = tileSize / 2;
    int rightIndex = leftIndex + elementsCount;    
    int startingOutput = leftIndex;
    mergeData(inputArray, outputArray, leftIndex, rightIndex, elementsCount, startingOutput);
}//end mergeSortNaiveStep2()

__device__ void mergeData(int* inputArray, int* outputArray, int leftIndex, int rightIndex, int elementCount, int startingOutput)
{
    
    int i = 0, j = 0;
    int k = startingOutput;

    while (i < elementCount && j < elementCount)
    {
        if (inputArray[leftIndex + i] <= inputArray[rightIndex + j])
        {
            outputArray[k] = inputArray[leftIndex + i];
            i++;
        }//end if left is less than right elements

        else
        {
            outputArray[k] = inputArray[rightIndex + j];
            j++;
        }//end if right is less than left elements

        k++;
    }//end while both have elements

    while (i < elementCount)
    {
        outputArray[k]  = inputArray[leftIndex + i];
        k++;
        i++;
        
    }//end while elements remain in left

    while (j < elementCount)
    {
        outputArray[k] = inputArray[rightIndex + j];
        k++;
        j++;
    }//end while elements remain in right

}//end mergeData()

__global__ void mergeSortBinary(int* inputArray, int* outputArray, int tileSize)
{
    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    int relativePositionA = globalThreadID % tileSize;
    int leftIndexA = (globalThreadID - relativePositionA) * 2;
    int globalPositionA = leftIndexA + relativePositionA;
    int globalPositionB = globalPositionA + tileSize;

    if (tileSize == 1)
    {
        leftIndexA = 2 * globalThreadID;
        globalPositionA = leftIndexA;
    }//end checking for single tile size. This means you are working on a thread and it's neighbor

    int valueA = inputArray[globalPositionA];
    int valueB = inputArray[globalPositionB];
    int rankA = rank(valueA, inputArray + leftIndexA + tileSize, tileSize, 1) + globalPositionA;//+ leftIndexA;// 
    int rankB = rank(valueB, inputArray + leftIndexA, tileSize, 0) + globalPositionA;//+ leftIndexA;// 

    outputArray[rankA] = valueA;
    outputArray[rankB] = valueB;
}//end mergeSortBinary()

__device__ int rank(int value, int* array, int tileSize, int upperSection)
{
    int left = 0;
    int right = tileSize;

    while (left < right)
    {
        int middle = (left + right) / 2;

        if (upperSection == 1)
        {
            if (array[middle] <= value)
            {
                left = middle + 1;
            }//end if value is further right

            else
            {
                right = middle;
            }//end if value was further left

        }//end if we are searching B section

        else
        {
            if (array[middle] < value)
            {
                left = middle + 1;
            }//end if value is further right

            else
            {
                right = middle;
            }//end if value was further left

        }//end if we are searching A section

    }//end while left and right have not converged

    return left;
}//end rank()