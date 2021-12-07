/*  David Cox
    GPU CSCD445
    Final Project
*/

#include "files.h"
#include <stdio.h>
#include <stdlib.h>

/*
	takes the file name, creates the file and pulls all of the input and 
    places it into the passed in array.
*/
void retrieveFileArray(const char* fileName, int* inputArray)
{
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("No input file found, try again.\n");
        exit(-1);
    }//end if no input file

    int i = 0;
    int j = 0;
    while (!feof (inputFile))
    {
        j = fscanf (inputFile, "%d", &inputArray[i]);
        i++;
    }//end while not end of file

    fclose(inputFile);
}//end retrieveFileArray

/*
	Counts how many lines are in the file so we know how big to make the 
    int array.
*/
int countFileLines(const char* fileName)
{
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("No input file found, try again.\n");
        exit(-1);
    }//end if no input file

    int lines = 0;
    char ch, previous;
    
    for (ch = fgetc(inputFile); ch != EOF; ch = fgetc(inputFile))
    {
        if (ch == '\n')
        {
            lines++;
            
        }//end if end of line
        
        previous = ch;
    }//end for each line that's not eof

    if (previous != '\n')
    {
        lines++;
    }//checking if EOF is on line with number.

    fclose(inputFile);
    return lines;
}//end countFileLines