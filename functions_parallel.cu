#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ void setup_kernel(curandState *state, unsigned long seed){
	if(threadIdx.x == 0){
		for(int p = 0; p < 2; p++){
			curand_init(seed, 1234 + blockIdx.x + p * blockDim.x, 0, &state[blockIdx.x + p * blockDim.x]);
		}
	}
}

__global__ void pipeline1(char *data, int data_num_cols, int data_num_rows, int motif_length, curandState* globalState, 
	char *collection, int *score){
	extern __shared__ char shared[];
	float *excKmerScores      = (float*)&shared;
	float *PDF = (float*)&excKmerScores[data_num_cols - motif_length + 1];
	char  *private_collection = (char*) &PDF[data_num_cols - motif_length + 2];


	__shared__ float private_profile[4*K_mer];
	__shared__ int private_score;
	__shared__ float private_ACGT_BG[4];
	__shared__ int motifIndex;
	__shared__ char replacementMotif[K_mer];
	__shared__ int converged;
	__shared__ float motifSample;
	
	int rounds;

	char mask[4] = {'A', 'C', 'G', 'T'};
	int col_mask_count[4] = {};

	int col;
	int row;
	int mask_index;
	int loop_in_col = (motif_length - 1)/blockDim.x + 1;
	int col_score;
	int temp;
	int random_col;
	float curPosScore = 1.0;
	int motifChoice;
	int motifChoice_new;

	// initialize covergence indicator
	if (threadIdx.x == 0) {
		converged = 0;

	}
	__syncthreads();


	for(rounds = 0; rounds < loops && converged < motif_length; rounds++){
	// for (rounds = 0; rounds < loops; rounds++) {

		if (rounds == 0) {

			// get the picked row index, the random int among threads in the block is the same since using the identical random state
			motifChoice = curand(&globalState[blockIdx.x]) % data_num_rows;

			__syncthreads();

			// Get the initial collection and an auxillary variable col_mask_count, which is the frequency of 4 chars within one column
			for (int l = 0; l < loop_in_col; l++) {

				col = threadIdx.x + l * loop_in_col;

				if (col < motif_length) {

					col_mask_count[0] = 0;
					col_mask_count[1] = 0;
					col_mask_count[2] = 0;
					col_mask_count[3] = 0;


					for (row = 0; row < data_num_rows; row++) {

						random_col = curand(&globalState[blockIdx.x]) % (data_num_cols - motif_length + 1);

						private_collection[col + row * motif_length] = data[col + random_col + row * data_num_cols];

						for (mask_index = 0; mask_index < 4; mask_index++) {

							if (private_collection[col + row * motif_length] == mask[mask_index]) {

								col_mask_count[mask_index]++;

								if (row != motifChoice) {

									atomicAdd(&(private_profile[col + mask_index * motif_length]), 1.0);
								}
							}
						}
					}


					// Get the initial column score
					col_score = data_num_rows - col_mask_count[0];

					for (int p = 1; p < 4; p++) {

						temp = data_num_rows - col_mask_count[p];

						if (temp < col_score) {

							col_score = temp;
						}
					}

					atomicAdd(&private_score, col_score);


					// Get the initial profile matrix
					for (row = 0; row < 4; row++) {

						private_profile[col + row * motif_length] = (private_profile[col + row * motif_length] + 1.0) / (4.0 + data_num_rows - 1.0);
					}
				}
			}
			__syncthreads();
		}
		


		// background probability, the probability of chars in the rows of the original dataset
		for(int l = 0; l < (data_num_cols - 1)/blockDim.x + 1; l++){

			col = threadIdx.x + l * blockDim.x;

			if(col < data_num_cols){

				for(int p = 0; p < 4; p++){

					if(data[col + motifChoice * data_num_cols] == mask[p]){

						atomicAdd(&(private_ACGT_BG[p]), 1.0/float(data_num_cols));
					}
				}	
			}
		}
		__syncthreads();



		// Scoring every k-mer possible in the excluded string using a "sliding window".
		int slide_w = data_num_cols - motif_length + 1;

		loop_in_col = (data_num_cols - 1) / blockDim.x + 1;

		for(int l = 0; l < loop_in_col; l++){

			col = threadIdx.x + l * blockDim.x;

			curPosScore = 1.0;

			__syncthreads();

			if(col < slide_w){

				for(int k = 0; k < motif_length; k++){

					for (int p = 0; p < 4; p++){

						if (data[col + k + motifChoice * data_num_cols] == mask[p]){

							curPosScore *= (private_profile[k + p * motif_length] / private_ACGT_BG[p]);
						}
					}
				}

				excKmerScores[col] = curPosScore;
			}	
		}
		__syncthreads();



		// Running sum to derive CDF:
		loop_in_col = (slide_w - 1)/blockDim.x + 1;


		for(int l = 0; l < loop_in_col; l++){

			col = threadIdx.x + l * blockDim.x;

			if(col < slide_w){

				for(int p = 0; p < col+1; p++){


					PDF[col + 1] += excKmerScores[p];
				}
			}
		}
		__syncthreads();



		// Generate a float value to choose a random column position
		if(threadIdx.x == 0){

			motifSample = curand_uniform(&globalState[blockIdx.x + blockDim.x]) * float(PDF[slide_w]);

			motifIndex = slide_w - 1;

			PDF[0] = 0;
		}
		__syncthreads();



		loop_in_col = (slide_w - 1)/blockDim.x + 1;

		for(int l = 0; l < loop_in_col; l++){

			col = threadIdx.x + l * blockDim.x;

			if(col < slide_w){

				if(motifSample <= PDF[col + 1] && motifSample > PDF[col]){

					motifIndex = col;
				}
			}
		}



		// Compare the new chosen motif with the old one
		// reset convegence indicator in each round

		if (threadIdx.x == 0) {

			converged = 0;
		}
		__syncthreads();



		// examine stop condition

		loop_in_col = (motif_length - 1) / blockDim.x + 1;

		for(int l = 0; l < loop_in_col; l++){

			col = threadIdx.x + l * blockDim.x;
			
			if(col < motif_length){

				replacementMotif[col] = data[motifIndex + col + motifChoice * data_num_cols];

				if(replacementMotif[col] == private_collection[col + motifChoice * motif_length]){

					atomicAdd(&converged, 1);
				}
			}
		}
		__syncthreads();


		motifChoice_new = curand(&globalState[blockIdx.x]) % data_num_rows;


		// replace the motif in the collction by the sampled one

		if(converged >= 0){

			for(int l = 0; l < loop_in_col; l++){

				col = threadIdx.x + l * blockDim.x;

				if (threadIdx.x == 0) {

					private_score = 0;
				}
				__syncthreads();


				if(col < motif_length){
					
					// updating private profile matrix
					for (mask_index = 0; mask_index < 4; mask_index++) {

						// removing values contributed by the old motif rows
						if (private_collection[col + motifChoice_new * motif_length] == mask[mask_index]) {

							col_mask_count[mask_index]--;

							private_profile[col + mask_index * motif_length] -= 1.0 / float(data_num_rows + 3.0);
						}

						// adding values contributed by the new motif rows
						if (replacementMotif[col] == mask[mask_index]) {

							col_mask_count[mask_index]++;

							private_profile[col + mask_index * motif_length] += 1.0 / float(data_num_rows + 3.0);
						}
					}


					// updating column score
					col_score = data_num_rows - col_mask_count[0];

					for (mask_index = 1; mask_index < 4; mask_index++) {

						temp = data_num_rows - col_mask_count[mask_index];

						if (temp < col_score) {

							col_score = temp;
						}
					}

					atomicAdd(&private_score, col_score);


					// updating private collection
					private_collection[col + motifChoice * motif_length] = replacementMotif[col];

				}
			}
		}	

		motifChoice = motifChoice_new;
	}
	__syncthreads();



	// Output the collection we build
	for(int l = 0; l < loop_in_col; l++){

		col = threadIdx.x + l * blockDim.x;

		if(col < motif_length){

			for(row = 0; row < data_num_rows; row++){

				collection[col + (row +  blockIdx.x * data_num_rows) * motif_length] = private_collection[col + row * motif_length];
			}
		}
		__syncthreads();
	}
	__syncthreads();



	// Output score for the current score
	if(threadIdx.x == 0){
		
		score[blockIdx.x] = private_score;
	}



	__syncthreads();
}



void pipelines(char *data_device, int data_num_cols, int data_num_rows, int pipeline_num, int motif_length, char *motifs_device){
	curandState *devStates;
	char *collection_device;
	int *scores;
	int *scores_device;
	float average_score = 0.0;
	cudaError_t cudaStatus;

	scores = (int *)malloc(sizeof(int)*pipeline_num);

	cudaStatus = cudaMalloc((void **)&devStates, sizeof(curandState) * pipeline_num * 2);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc for devStates failed!\n");
	}

	cudaStatus = cudaMalloc((void **)&collection_device, sizeof(char) * motif_length * data_num_rows * pipeline_num);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc for collection_device failed!\n");
	}

	cudaStatus = cudaMalloc((void **)&scores_device, sizeof(int) * pipeline_num);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc for score failed!\n");
	}


	clock_t begin_compute = clock();

	srand(time(0));

	int seed = rand();
	// int seed = 123;
	int char_size = sizeof(char) * data_num_rows * motif_length;
	int float_size = sizeof(float) * ((data_num_cols - motif_length + 1) * 2 + 1);

	int shared_memory = char_size + float_size;


	dim3 gridDim(pipeline_num);
	// pipeline_num is the defined block numbers we want to launch

	dim3 blockDim(160);
	// block size is the fixed number of threads launched in one block

	// set up states in each random number generator
	setup_kernel<<<gridDim, blockDim>>>(devStates, seed);

	// kernel pipeline1 is the kernel to generate sampled collections
	pipeline1<<<gridDim, blockDim, shared_memory>>>(data_device, data_num_cols, data_num_rows, 
													motif_length, devStates, motifs_device, scores_device);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaStatus = cudaMemcpy(scores, scores_device, pipeline_num * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy for output_host D2H failed!\n");
	}

	cudaFree(devStates);
	cudaFree(collection_device);
	cudaFree(scores_device);

	clock_t end_compute = clock();
	double elapsed_secs_compute_parallel = double(end_compute - begin_compute) / CLOCKS_PER_SEC;
	printf("**Parallel: Computing time taken = %.9f\n", elapsed_secs_compute_parallel);

	for (int i = 0; i < pipeline_num; i++) {
		average_score += float(scores[i]) / pipeline_num;
	}
	
	printf("  The average score of output collections is %.2f.\n", average_score);

	free(scores);
}