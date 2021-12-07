/*  David Cox
    GPU CSCD445
    Final Project
*/

#include <stdio.h>
#include <stdlib.h>
#include "mergeCPU.h"

void mergeSortCPU(int array[], int left, int right)
{
    if (left < right)
    {
        int middle = (left + right - 1) / 2;
        mergeSortCPU(array, left, middle);
        mergeSortCPU(array, middle + 1, right);

        mergeCPU(array, left, middle, right);
    }//end if left passes right

}//end mergeSort()


//merging 2 sub arrays. Left to middle, and middle+1 to right
void mergeCPU(int array[], int left, int middle, int right)
{
    int i, j, k;    //temp ints for looping
    int elements1 = middle - left + 1;
    int elements2 = right - middle;
    size_t leftBytes = elements1 * sizeof(int);
    size_t rightBytes = elements2 * sizeof(int);
    int* leftArray = (int*) malloc(leftBytes);
    int* rightArray = (int*) malloc(rightBytes);

    for (i = 0; i < elements1; i++)
    {
        leftArray[i] = array[left + i];
    }//end fill leftArray

    for (i = 0; i < elements2; i++)
    {
        rightArray[i] = array[middle + 1 + i];
    }//end fill rightArray

    i = 0;
    j = 0;
    k = left;

    while (i < elements1 && j < elements2)
    {
        if (leftArray[i] <= rightArray[j])
        {
            array[k] = leftArray[i];
            i++;
        }//end if left < right

        else
        {
            array[k] = rightArray[j];
            j++;
        }//end else right < left

        k++;
    }//end while elements remain

    while (i < elements1)
    {
        array[k] = leftArray[i];
        k++;
        i++;
    }//end while elements remain in left

    while (j < elements2)
    {
        array[k] = rightArray[j];
        k++;
        j++;
    }//end while elements remain in right

    free(leftArray);
    free(rightArray);

}//end merge