// do not modify this file
// call each kernel implemented in the kernel.cu
// generates timing info
// tests for functional verification

#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include "kernel.cu"
#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

void histogram(unsigned int *input, unsigned int *bins,
               unsigned int num_elements, unsigned int num_bins, int kernel_version) {


 if (kernel_version == 0) {
  // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(512), gridDim(30);
    histogram_global_kernel<<<gridDim, blockDim, num_bins * sizeof(unsigned int)>>>(
        input, bins, num_elements, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Make sure bin values are not too large
  {
    dim3 blockDim(512);
    dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
    convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
 }
 else if (kernel_version==1) {
 // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(512), gridDim(30);
    histogram_shared_kernel<<<gridDim, blockDim,
                       num_bins * sizeof(unsigned int)>>>(
        input, bins, num_elements, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Make sure bin values are not too large
  {
    dim3 blockDim(512);
    dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
    convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
 }

else if (kernel_version==2) {
 // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(512), gridDim(30);
    histogram_shared_accumulate_kernel<<<gridDim, blockDim,
                       num_bins * sizeof(unsigned int)>>>(
        input, bins, num_elements, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Make sure bin values are not too large
  {
    dim3 blockDim(512);
    dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
    convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
 }


}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  int version; // kernel version global or shared 
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  cudaEvent_t astartEvent, astopEvent;
  float aelapsedTime;
  cudaEventCreate(&astartEvent);
  cudaEventCreate(&astopEvent);
  
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  CUDA_CHECK(cudaMalloc((void **)&deviceInput,
                        inputLength * sizeof(unsigned int)));
  CUDA_CHECK(
      cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceInput, hostInput,
                        inputLength * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  // wbTime_start(Compute, "Performing CUDA computation");

  version = atoi(argv[5]); 
  cudaEventRecord(astartEvent, 0);
  histogram(deviceInput, deviceBins, inputLength, NUM_BINS,version);
  // wbTime_stop(Compute, "Performing CUDA computation");

  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  printf("\n");
  printf("Total compute time (ms) %f for version %d\n",aelapsedTime,version);
  printf("\n");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // Verify correctness
  // -----------------------------------------------------
  printf ("running version %d\n", version);
  if (version == 0 )
     wbLog(TRACE, "Checking global memory only kernel");
  else if (version == 1) 
     wbLog(TRACE, "Launching shared memory kernel");
  else if (version == 2) 
     wbLog(TRACE, "Launching accumulator kernel");
  wbSolution(args, hostBins, NUM_BINS);

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceBins));
  wbTime_stop(GPU, "Freeing GPU Memory");


  free(hostBins);
  free(hostInput);
  return 0;
}
