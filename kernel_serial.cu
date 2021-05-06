#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <ctime>
#include <random>
#define K_mer 17
#define sample_num 200
#define loops 1000
#include "functions_serial.cu"

using namespace std;

void constructCollection(char *input, char *collection, int Num_seqs, int DNA_len);
void print_char_data(char *data, int row, int col);
void print_float_data(float *data, int row, int col);
void profile_matrix(float *profile, char *collection, char *mask, int rowNum, int excluded_line);
void background_prob(float *ACGT_BG, char *input, char *mask, int input_col, int random);
void score_sliding_window(float *excKmerScores, float *profile, float *ACGT_BG, char *input, char *mask, int DNA_len, int random);
bool convergence(char *replacementMotif, char *collection, int random, int col_len);
int score(char *data, char *mask, int data_row, int data_col);

int main() {
	char *input;      //Each DNA sequence is stored at a specific index in the array.
	char *collection;     //The current motif from each DNA strand is stored at each appropriate index in the array.
	char mask[4] = { 'A', 'C', 'G', 'T'};
	float *profile;            //Profile declaration
	int DNA_len = 0;
	// int bestScore = 0;
	int random = 0;
	float ACGT_BG[4];       //Array that holds background frequencies for each character in the excluded string.
	char replacementMotif[K_mer];             //Sampled motif to replace excluded motif in Gibb's Sampling Loop.
	float motifChoice = 0.0;                    //Initlaizing random number to use to sample Motif from created PDF.
	int motifIndex = 0;                             //Motif index that is chosen from distribution as mtoif to replace.
	bool converged = false;                     //Indicating Motif convergence.

	int rounds = 0;
	float *excKmerScores;
	float *PDF;
	int rowNum;
	int row;
	int output_i = 0;
	int curScore;
	float average_score = 0.0;

	ofstream outf;

	//DNA input file reading
	ifstream file("yst09.txt");

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

	printf("Total rows: %d \n", rowNum);

	printf("Number of total columns: %d \n", DNA_len);

	// Allocate memory space for these lines with the fixed length: DNA_len
	input = (char *)malloc(sizeof(char)*DNA_len*rowNum);
	

	// Read lines and store them in the allocated memory space
	ifstream infile("yst09.txt");

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
	

	// Allocate memory space for collction, profile matrix, scores derived by sliding window and PDF(Porbability Density Function
	collection = (char *)malloc(sizeof(char)*K_mer*rowNum);

	profile = (float *)malloc(sizeof(float) * 4 * K_mer);

	excKmerScores = (float *)malloc(sizeof(float)*(DNA_len - K_mer + 1));

	PDF = (float *)malloc(sizeof(float)*(DNA_len - K_mer + 1));


	// Create the Initial collection
	constructCollection(input, collection, rowNum, DNA_len);
	

	outf.open("serial_sampled.txt");

	for (int sample_i = 0; sample_i < sample_num; sample_i++) {

		std::mt19937 mt(sample_i);  // set a seed for the random sample generator
		std::uniform_int_distribution<int> dist(0, rowNum);  // set the distribution of random variable

		rounds = 0;

		// while (rounds < loops && !converged) {
		while (rounds < loops) {

			random = dist(mt);  //Selecting random motif to exclude from collection.

			profile_matrix(profile, collection, mask, rowNum, random);

			background_prob(ACGT_BG, input, mask, DNA_len, random);

			score_sliding_window(excKmerScores, profile, ACGT_BG, input, mask, DNA_len, random);

			for (int i = 0; i < DNA_len - K_mer + 1; i++) {

				if (i == 0) {

					PDF[i] = excKmerScores[i];
				}
				else {

					PDF[i] = PDF[i - 1] + excKmerScores[i];
				}
			}


			motifChoice = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (PDF[DNA_len - K_mer])));      //Get random (floating point) sample within index range.

			for (int i = 0; i < (DNA_len - K_mer + 1); i++) {                                                                 //Choosing motif index.

				if (motifChoice < PDF[i] && i == 0) {

					motifIndex = i;
				}
				else if (motifChoice < PDF[i] && motifChoice > PDF[i - 1]) {

					motifIndex = i;
				}
			}

			for (int i = 0; i < K_mer; i++) {

				replacementMotif[i] = input[i + motifIndex + random * DNA_len];
			}
			

			//Replacing j-th motif with replacement motif.
			converged = convergence(replacementMotif, collection, random, K_mer);

			rounds++;

		}

		
		curScore = score(collection, mask, rowNum, K_mer);

		average_score += float(curScore) / sample_num;
		

		for (output_i = 0; output_i < K_mer * rowNum; output_i++) {
			
			if (output_i % K_mer == 0 && output_i != 0) {

				outf << endl;
			}
			outf << collection[output_i];
		}
		outf << endl;
	}


	clock_t end = clock();

	double elapsed_secs_serial = double(end - begin) / CLOCKS_PER_SEC;

	outf.close();

	printf("**Serial: time taken = %.9f\n", elapsed_secs_serial);

	printf("  The average score of output collections is %.2f.\n", average_score);


	free(input);

	free(excKmerScores);

	free(PDF);

	return 0;
}


