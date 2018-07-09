#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#define PI 3.14159265358979323846

const int sinogramHeight = 367; // total parallel x-ray projections
const int sinogramWidth = 180; // total slices
const int filterLength = (sinogramHeight%2 == 0) ? (sinogramHeight+1)*2 + 1 : sinogramHeight*2 + 1;

// 39 ? 3 & 507  /  367 ? 367 & 367  /  729 ? 692 & 768
const int bpkernel_blocks = 367;
const int bpkernel_threads = 367;


/************************
* Methods declarations *
* **********************/

/* Read file and load to 2D array */
template<typename T, size_t cols>
void loadFromFile(std::string filename, T mat[][cols], size_t rows);

/* Save 2D array to file */
template<typename T, size_t cols>
void saveToFile(std::string filename, T mat[][cols], size_t rows);

/* build a Shepp Logan 1D filter */
void getFitler(float* a, int s);


/***************
* CUDA kernels *
* **************/

// Backprojection kernel
__global__ void k_backprojection(float *backProjection, float *sinogram) {
	// thread's local index
	int tx = threadIdx.x;

	// thread's global index
	int bpIdx = blockIdx.x * blockDim.x + tx;

	const int bpWidth = sinogramHeight; // size of the back projection image
	const int totalSlices = sinogramWidth;

	// variables shared within each block
	__shared__ int midindex;
	__shared__ float angles[totalSlices]; // the angles at which the slices were taken (rads)

	// only 1 thread per block should initialize shared variables
	if (tx == 0) {
		// find middle index of the projections
		midindex = int(bpWidth / 2);
	} 
	// // meanwile other threads will compute the angles
	else if ((tx-1) < totalSlices) {
		angles[(tx-1)] = (tx-1) * PI / totalSlices;
	}

	// ensure all inits completed before continuing
	__syncthreads();


	// equivalent of the index in row-column
	int r = int(bpIdx/bpWidth);
	int c = bpIdx - (r * bpWidth);

	// local accumulator for the backprojection
	float val = 0;

	for (int sliceIdx = 0; sliceIdx<totalSlices; sliceIdx++) {
		// cartesian coordinates (x,y) for current pixel (cause some geometry is needed)
		int x = c - midindex;
		int y = -(r - midindex);

		// find where the projection of this pixel ended up in this sinogram's slice
		int distance = (int)round(x*cos(angles[sliceIdx]) + y * sin(angles[sliceIdx]));

		// convert to array index system
		int projectionIdx = midindex + distance;

		// check if we are in boundaries
		if ((projectionIdx > -1) && (projectionIdx < bpWidth)) {
			val += sinogram[projectionIdx * sinogramWidth + sliceIdx] / totalSlices;
		}
	}

	// update in global memory
	backProjection[bpIdx] = val;

}

// Sinogram filtering kernel
__global__ void k_filtersinogram(float *sinogram, float *fltr) {
	// thread's local index
	int tx = threadIdx.x;

	// block index (corresponding to the ith slice)
	int bx = blockIdx.x;

	const int sliceLength = sinogramHeight;

	__shared__ float slice[sliceLength];
	__shared__ float filter[filterLength];

	// load slice to shared memory
	slice[tx] = sinogram[tx * sinogramWidth + bx];

	// load filter to shared memory
	filter[tx] = fltr[tx];
	filter[tx+sliceLength+1] = fltr[tx+sliceLength+1]; // each thread handles 2 elements
	if (tx == 0) {
		filter[sliceLength] = fltr[sliceLength]; // handle the middle element with thread#0
	}

	// wait for all threads to load data
	__syncthreads();

	// convolve slice with filter (centered convolution, eq. to matlab's conv(a,b,'same'))
	float val = 0;
	for (int j=0; j<sliceLength; j++) {
		val +=  slice[(sliceLength-1)-j] * filter[tx+j+1];
	}

	// update in global memory
	sinogram[tx * sinogramWidth + bx] = val;	
}

// Sinogram slice filtering kernel - testing purpose only
__global__ void k_filterslice(float *sinogram, float *filter, int colIdx) {
	const int len = sinogramHeight;

	// thread's local index
	int tx = threadIdx.x;

	__shared__ float slice[len];
	
	slice[tx] = sinogram[tx * sinogramWidth + colIdx];
	
	__syncthreads();

	float val = 0;
	for (int j=0; j<len; j++) {
		val +=  slice[(len-1)-j] * filter[tx+j+1];
	}

	sinogram[tx * sinogramWidth + colIdx] = val;	
}


/**************************
* Program starting point *
**************************/

cudaEvent_t start, stop;
float duration1 = 0, duration2 = 0;

int main(int argc, char ** argv) {
	// where to read and write data
	std::string inputFile = "sinogram.txt";
	std::string outputFile = "reconstructed.txt";

	// load image in 2D array
	float h_sinogram[sinogramHeight][sinogramWidth]; // input
	loadFromFile(inputFile, h_sinogram, sinogramHeight);

	// build the filter
	float h_filter[filterLength];
	getFitler(h_filter, filterLength);

	// back-projection matrix
	const int bpWidth = sinogramHeight; // number of parallel x-ray emitters/receivers
	float h_backProjection[bpWidth][bpWidth];


	/********************
	* Start CUDA stuff *
	********************/
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// total bytes needed for each matrix
	size_t FILTER_BYTES = sizeof(float) * filterLength;
	size_t SINOGRAM_BYTES = sizeof(float) * sinogramHeight * sinogramWidth;
	size_t BACKPROJECTION_BYTES = sizeof(float) * bpWidth * bpWidth;

	// declare GPU memory pointers
	float *d_filter;
	float *d_sinogram;
	float *d_backProjection;

	// allocate GPU memory
	cudaMalloc((void**)&d_filter, FILTER_BYTES);
	cudaMalloc((void**)&d_sinogram, SINOGRAM_BYTES);
	cudaMalloc((void**)&d_backProjection, BACKPROJECTION_BYTES);

	// copy the inputs (filter and sinogram) to GPU memory
	cudaMemcpy(d_filter, h_filter, FILTER_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sinogram, h_sinogram, SINOGRAM_BYTES, cudaMemcpyHostToDevice);

	
	// launch the filtering kernel
	cudaEventRecord(start);

	k_filtersinogram << < sinogramWidth, sinogramHeight >> >(d_sinogram, d_filter);

	// for (int colIdx=0; colIdx<sinogramWidth; colIdx++) {
	// 	k_filterslice << < 1, sinogramHeight >> >(d_sinogram, d_filter,colIdx);
	// }
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration1, start, stop);


	// launch the backprojection kernel
	cudaEventRecord(start);

	k_backprojection << < bpkernel_blocks, bpkernel_threads >> >(d_backProjection, d_sinogram);

	cudaEventRecord(stop);

	// copy back the result to CPU memory
	cudaMemcpy(h_backProjection, d_backProjection, BACKPROJECTION_BYTES, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration2, start, stop);

	// free device memory
	cudaFree(d_filter);
	cudaFree(d_sinogram);
	cudaFree(d_backProjection);

	/******************
	* End CUDA stuff *
	******************/


	// save to file
	saveToFile(outputFile, h_backProjection, bpWidth);

	/* To visualize in matlab: im = load("reconstructed.txt"); imshow(im,[]); */

	printf("Filtering kernel time = %f ms\n", duration1);
	printf("Backprojection kernel time = %f ms\n", duration2);

	return 0;
}


/**********************
* Methods definition *
**********************/

template<typename T, size_t cols>
void loadFromFile(std::string filename, T mat[][cols], size_t rows) {
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


template<typename T, size_t cols>
void saveToFile(std::string filename, T mat[][cols], size_t rows) {
	std::ofstream fout;
	fout.open(filename.c_str());
	fout << "[ ";
	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<cols; j++) {
			fout << mat[i][j];
			if (j<cols - 1) { 
				fout << ", ";
			}
			else if (i<rows - 1) {
				fout << "; ";
			}
		}
	}
	fout << " ]";
	fout.close();
}

void getFitler(float* a, int len) {
    int n = int(-len/2);
    for (int i=0; i<len; i++) {
        a[i] = -2. / (pow(PI,2.) * (4 * pow(n+i,2.) - 1) );
    }
}
