# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = main.o mergeGPU.o
OBJS = mergeCPU.o files.o #timing.o

# make and compile
project:$(OBJS) $(GPUOBJS)
	$(NVCC) -arch=sm_37 -o project $(OBJS) $(GPUOBJS) 

mergeCPU.o: mergeCPU.c
	$(CXX) -c mergeCPU.c
#timing.o: timing.c
#	$(CXX) -c timing.c
files.o: files.c
	$(CXX) -c files.c
mergeGPU.o: mergeGPU.cu
	$(NVCC) -arch=sm_37 -c mergeGPU.cu
main.o: main.cu
	$(NVCC) -arch=sm_37 -c main.cu 

clean:
	rm -f *.o
	rm -f project
	rm CPUoutput.txt
	rm GPUNaiveoutput.txt
	rm GPUBinaryoutput.txt
