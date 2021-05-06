#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <ctime>
#include <random>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


#define K_mer 17
#define seed_num 10
#define loops 500


#include "functions_serial.cu"
#include "functions_parallel.cu"

using namespace std;

void print_char_data(char *data, int row, int col);
void print_float_data(float *data, int row, int col);

void pipelines(char *data_host, int data_num_cols, int data_num_rows, int pipeline_num, int motif_length, char *output_device);

int main() {
	char *input;      //Each DNA sequence is stored at a specific index in the array. 
	int DNA_len = 0;
	// char mask[4] = { 'A', 'C', 'G', 'T' };
	char *motifs_return;
	cudaError_t cudaStatus;
	char *input_device;
	char *motifs_return_device;
	int rowNum;
	int row;


	//DNA input file reading
	ifstream file("datasets/cleaned/real/yst09r.txt");
	string inputLineStr;

	// get how many lines will be taken in the algorithm
	rowNum = 0;
	while (getline(file, inputLineStr)) {
		if (inputLineStr.at(0) != '>') {
			rowNum++;
			DNA_len = inputLineStr.length();
		}
	}
	file.close();
	printf("Number of total rows: %d \n", rowNum);
	printf("Number of total columns: %d \n", DNA_len);

	// Allocate memory space for these lines with the fixed length: DNA_len
	input = (char *)malloc(sizeof(char)*DNA_len*rowNum);
	motifs_return = (char *)malloc(sizeof(char) * K_mer * rowNum * seed_num);


	// Read lines and store them in the allocated memory space
	ifstream infile("datasets/cleaned/real/yst09r.txt");
	std::cout << "\nLoading data...\n";
	row = 0;
	int offset = 0;
	while (getline(infile, inputLineStr)) {               //Extracting file line by line.
		if (inputLineStr.at(0) != '>') {                  //Only considers DNA strings, NOT sequence numbers.
			if (row != 0) {
				offset += DNA_len;
			}
			memcpy(input + offset, inputLineStr.c_str(), sizeof(char)*DNA_len);
			row++;
		}
	}
	infile.close();
	printf("Loaded.\n");


	clock_t begin = clock();

	cudaStatus = cudaMalloc((void **)&input_device, sizeof(char) * DNA_len * rowNum);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc for input_device failed!\n");
	}

	cudaStatus = cudaMalloc((void **)&motifs_return_device, sizeof(char) * K_mer * rowNum * seed_num);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc for output_device failed!\n");
	}

	cudaStatus = cudaMemcpy(input_device, input, sizeof(char) * DNA_len * rowNum, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy for input_device H2D failed!\n");
	}


	pipelines(input_device, DNA_len, rowNum, seed_num, K_mer, motifs_return_device);
	

	cudaStatus = cudaMemcpy(motifs_return, motifs_return_device, K_mer * rowNum * seed_num * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy for output_host D2H failed!\n");
	}
	clock_t end = clock();

	double elapsed_secs_parallel = double(end - begin) / CLOCKS_PER_SEC;


	printf("**Parallel: Total time taken = %.9f\n", elapsed_secs_parallel);

	ofstream outf;
	outf.open("datasets/sampled_parallel.txt");
	
	
	for (int i = 0; i < K_mer * rowNum * seed_num; i++) {
		if (i % K_mer == 0 && i != 0) {
			outf << endl;
		}
		outf << motifs_return[i];
	}
	
	outf.close();


	cudaFree(input_device);
	cudaFree(motifs_return_device);
	free(input);
		
	return 0;
}


