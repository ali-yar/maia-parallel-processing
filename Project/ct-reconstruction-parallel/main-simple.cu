#include <iostream>
#include <math.h>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define PI 3.14159265358979323846
#define SINO_HEIGHT 367
#define SINO_WIDTH 90

__global__ void kernel(float OUT[SINO_HEIGHT][SINO_HEIGHT], float IN[SINO_HEIGHT][SINO_WIDTH]) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	// find the middle index of the projections
	int midindex = int(SINO_HEIGHT / 2);
	
	int step = 180 / SINO_WIDTH;

	float val = 0;
	for (int t = 0; t<SINO_WIDTH; t++) {
		float d = (t * step) * (PI / 180.); // angle 
		int x = c - midindex;
		int y = r - midindex;
		int rotCoords = round(midindex + x * sin(d) + y * cos(d));

		if ((rotCoords > -1) && (rotCoords < SINO_HEIGHT)) {
			val += IN[rotCoords][t] / SINO_WIDTH;
		}
	}
	OUT[r][c] = val;
}

template<typename T, size_t cols>
void getData(std::string filename, T mat[][cols], size_t rows);

int main(int argc, char ** argv) {
	std::string inputFile = "sinogram.txt";
	std::string outputFile = "reconstructed.txt";

	// load image in 2D array
	float h_sino[SINO_HEIGHT][SINO_WIDTH]; // input
	getData(inputFile, h_sino, SINO_HEIGHT);

	const int totalProjections = SINO_HEIGHT;
	const int totalAngles = SINO_WIDTH;

	int step = 180 / totalAngles;

	float thetas[totalAngles];
	for (int i = 0; i<totalAngles; i++) {
		thetas[i] = (i * step) * (PI / 180.);
	}

	float h_img[totalProjections][totalProjections]; // output

	// *************** START CUDA stuff ***************

	int SINO_BYTES = sizeof(float) * SINO_HEIGHT * SINO_WIDTH;
	int IMG_BYTES = sizeof(float) * totalProjections * totalProjections;

	// declare GPU memory pointers
	float (*d_sino)[SINO_WIDTH];
	float (*d_img)[totalProjections];

	// allocate GPU memory
	cudaMalloc((void**)&d_sino, SINO_BYTES);
	cudaMalloc((void**)&d_img, IMG_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_sino, h_sino, SINO_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	dim3 blocks(20, 20);
	dim3 threads(18, 18);
	kernel <<< blocks, threads >>>(d_img, d_sino);

	// copy back the result array to the CPU
	cudaMemcpy(h_img, d_img, IMG_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_sino);
	cudaFree(d_img);

	// *************** END CUDA stuff ***************

	// save to file
	std::ofstream fout;
	fout.open(outputFile.c_str());
	fout << "[ ";
	for (int i = 0; i<totalProjections; i++) {
		for (int j = 0; j<totalProjections; j++) {
			fout << h_img[i][j];
			if (j<totalProjections - 1) {
				fout << ", ";
			}
			else if (i<totalProjections - 1) {
				fout << "; ";
			}
		}
	}
	fout << " ]";
	fout.close();

	return 0;
}

template<typename T, size_t cols>
void getData(std::string filename, T mat[][cols], size_t rows) {
	std::ifstream fin;
	fin.open(filename.c_str());
	if (!fin) { std::cerr << "cannot open file"; }
	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<cols; j++) {
			fin >> mat[i][j];
		}
	}
	fin.close();
}
