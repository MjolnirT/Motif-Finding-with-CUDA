void constructCollection(char *input, char *collection, int Num_seqs, int DNA_len) {
	std::mt19937 mt(rand());
	std::uniform_int_distribution<int> dist(0, DNA_len - K_mer + 1);

	for (int row = 0; row < Num_seqs; row++) {
		int randPos = dist(mt);
		for (int col = 0; col < K_mer; col++) {
			collection[col + row * K_mer] = input[randPos + col + row * DNA_len];
		}
		// printf("%d ", randPos);
	}
}

void print_char_data(char *data, int row, int col) {
	printf("\n");
	for (int i = 0; i < row; i++) {
		printf("Row %d:  ", i);
		for (int j = 0; j < col; j++) {
			printf("%c", data[j + i * col]);
		}
		printf("\n");
	}
}

void print_float_data(float *data, int row, int col) {
	printf("\n");
	for (int i = 0; i < row; i++) {
		printf("Row %d:  ", i);
		for (int j = 0; j < col; j++) {
			printf("%.4f  ", data[j + i * col]);
		}
		printf("\n");
	}
}

int score(char *data, char *mask, int data_row, int data_col) {
	int score = 0;
	int mask_count[4];
	int maxNum = 0;
	int colScore = 0;

	for (int col = 0; col < data_col; col++) {

		for (int k = 0; k < 4; k++) {
			mask_count[k] = 0;
		}

		maxNum = 0;

		for (int row = 0; row < data_row; row++) {
			for (int mask_p = 0; mask_p < 4; mask_p++) {
				if (data[col + row * data_col] == mask[mask_p]) {
					mask_count[mask_p]++;
				}
			}
		}

		for (int mask_p = 0; mask_p < 4; mask_p++) {
			if (mask_p == 0) {
				maxNum = mask_count[mask_p];
			}
			else if (mask_count[mask_p] > maxNum) {
				maxNum = mask_count[mask_p];
			}
		}

		colScore = data_row - maxNum;
		printf("colScore add %2d.\n", colScore);
		score += colScore;
	}
	printf("\n\n\n");
	return score;
}

void profile_matrix(float *profile, char *collection, char *mask, int rowNum, int excluded_line) {

	for (int col = 0; col < K_mer; col++) {

		for (int row = 0; row < 4; row++) {
			profile[col + row * K_mer] = 0; // initializing profile matrix in the column
		}

		for (int row = 0; row < rowNum; row++) { // count the frequency of each word in the column
			if (row != excluded_line) {
				for (int p = 0; p < 4; p++) {
					if (collection[col + row * K_mer] == mask[p]) {
						profile[col + p * K_mer] += 1;
					}
				}
			}
		}

		for (int p = 0; p < 4; p++) {
			// printf("p: %d, column %d has counts: %.2f \n", p, col, profile[col + p * K_mer]);
			profile[col + p * K_mer] = (float)(1 + profile[col + p * K_mer]) / (float)(4 + rowNum - 1);
		}
	}
}

void background_prob(float *ACGT_BG, char *input, char *mask, int input_col, int random) {
	for (int i = 0; i < 4; i++) {   //Initializing background array.
		ACGT_BG[i] = 0.0;
	}

	for (int col = 0; col < input_col; col++) {
		for (int p = 0; p < 4; p++) {
			if (input[col + random * input_col] == mask[p]) {
				ACGT_BG[p] += 1;
			}
		}
	}

	for (int i = 0; i < 4; i++) {          //Divides letter coubt by total DNA string length to get final background probabilities.
		ACGT_BG[i] = ACGT_BG[i] / input_col;
	}
}


void score_sliding_window(float *excKmerScores, float *profile, float *ACGT_BG, char *input, char *mask, int DNA_len, int random) {
	float curPosValue = 0.0;
	for (int i = 0; i < (DNA_len - K_mer + 1); i++) {                    //Loops through every possible sub-string (k-mer).
		float curPosScore = 1.0;                                        //Initializing score for k-mer at current index. (Multiplied value)
		for (int j = 0; j < K_mer; j++) {                               //Loops through individual k-mer to score.
			for (int p = 0; p < 4; p++) {
				if (input[random*DNA_len + i + j] == mask[p]) {
					curPosValue = profile[p*K_mer + j] / ACGT_BG[p];  //prob(generating motif from profile matrix) / prob(generating motif from background(DNA String))
				}
			}

			curPosScore *= curPosValue;                                                 //curPosScore = curPosScore * curPosValue    (Running Product)

		}

		excKmerScores[i] = curPosScore;                                                 //Store k-mer score for that particular index in the DNA string.

	}
}

bool convergence(char *replacementMotif, char *collection, int random, int col_len) {
	bool notreplaceIndicator = true;
	for (int i = 0; i < col_len; i++) {
		if (replacementMotif[i] != collection[random*col_len + i]) {
			notreplaceIndicator = false;
		}
	}

	if (!notreplaceIndicator) {
		for (int i = 0; i < col_len; i++) {
			collection[random*col_len + i] = replacementMotif[i];
		}
	}
	else {
		return true;
	}

	return false;

}