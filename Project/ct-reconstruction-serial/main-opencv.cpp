#include <iostream>
#include <fstream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.14159265358979323846

const int sinogramHeight = 367;
const int sinogramWidth = 90;


/************************
 * Methods declarations *
 * **********************/

/* Read file and load to 2D array */
template<typename T, size_t cols>
void loadFromFile(std::string filename, T mat[][cols], size_t rows);

/* Save 2D array to file */
template<typename T, size_t cols>
void saveToFile(std::string filename, T mat[][cols], size_t rows);

/* Convert 2D array to OpenCV Mat */
template<typename T, size_t cols>
void convertToMat(T mat[][cols], cv::Mat &dst, size_t rows, int type, bool normalize=false);

/* Display matrix as image */
template<typename T, size_t cols>
void cvimshow(T mat[][cols], size_t rows, bool flip=false, std::string title="image", int waitKey=0);


/**************************
 * Program starting point *
 **************************/

int main() {
    // where to read and write data
    std::string inputFile = "sinogram.txt";
    std::string outputFile = "reconstructed.txt";

    // load sinogram
    float sinogram[sinogramHeight][sinogramWidth];
    loadFromFile(inputFile, sinogram, sinogramHeight);

    // display the sinogram
    cvimshow(sinogram, sinogramHeight, false, "sinogram");

    // number of parallel x-ray emitters/receivers
    const int totalProjections = sinogramHeight;
    // number of slices (taken at different angles within the range [0-179])
    const int totalSlices  = sinogramWidth;

    // the angle increment between every 2 slices
    int step = 180 / totalSlices;

    // angles at which the slices were taken (min 0°, max 179°)
    double angles[totalSlices];
    for (int i=0; i<totalSlices; i++) {
        angles[i] = (i * step) * (PI/180.);
    }

    // back-projection matrix initialized to 0
    const int bpWidth = totalProjections;
    float backProjection[bpWidth][bpWidth] = {0.0};

    // find the middle index of the projections
    int midindex = int(bpWidth/2);

    // reconstruct pixel by pixel
    for (int r=0; r<bpWidth; r++) {
        for (int c=0; c<bpWidth; c++) {
            // find the contribution of each slice to the reconstruction matrix
            for (int sliceIdx=0; sliceIdx<totalSlices; sliceIdx++) {
                // the angle at which the slice was taken
                double angle = angles[sliceIdx];

                // cartesian coordinates (x,y) for current pixel (cause some geometry is needed)
                int x = c - midindex;
                int y = - (r - midindex);
                /* e.g. if matrix/image is 367x367 and midindex = 367/2 = 183
                then (row,col)=(0,0) corresponds to (x,y)=(-183,183)
                and  (row,col)=(183,183) corresponds to (x,y)=(0,0) */

                // find where the projection of this pixel ended up in this sinogram's slice
                int distance = (int) round(x*cos(angle) + y*sin(angle));
                /* The geometry is done in cartesian system, so it is the distance from center of sinogram.
                The geometry here is about the rotation matrix which gives the relationship between
                the source coordinates and the destination coordinates, i.e. x' = x*cos(theta) + y*sin(theta)
                and y' = x*-sin(theta) + y*cos(theta). Here we only need the x'. */

                // convert to array index system
                int projectionIdx = midindex + distance;

                // check if we are in boundaries
                if ((projectionIdx > -1) && (projectionIdx < bpWidth)) {
                    // update the pixel's value
                    backProjection[r][c] += sinogram[projectionIdx][sliceIdx]/totalSlices;
                }
            }
        }
    }

    // save to file
    saveToFile(outputFile, backProjection, bpWidth);
    /* To visualize in matlab: im = load("reconstructed.txt"); imshow(im,[]); */

    // display the reconstruction
    cvimshow(backProjection, bpWidth, false, "reconstructed");

    return 0;
}


/**********************
 * Methods definition *
 **********************/

template<typename T, size_t cols>
void loadFromFile(std::string filename, T mat[][cols], size_t rows) {
    std::ifstream fin;
    fin.open(filename);
    if (!fin) { std::cerr << "cannot open file"; }
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            fin >> mat[i][j];
        }
    }
    fin.close();
}


template<typename T, size_t cols>
void saveToFile(std::string filename, T mat[][cols], size_t rows) {
    std::ofstream fout;
    fout.open(filename);
    fout << "[ ";
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            fout << mat[i][j];
            if (j<cols-1) {
                fout << ", ";
            } else if (i<rows-1) {
                fout << "; ";
            }
        }
    }
    fout << " ]";
    fout.close();
}

template<typename T, size_t cols>
void convertToMat(T mat[][cols], cv::Mat &dst, size_t rows, int type, bool normalize) {
    dst.create(rows,cols,type);
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            dst.at<float>(i,j) =  mat[i][j];
        }
    }
    if (normalize) {
        cv::normalize(dst,dst,0,255,CV_MINMAX,CV_8U);
    }
}

template<typename T, size_t cols>
void cvimshow (T mat[][cols], size_t rows, bool flip, std::string title, int waitKey) {
    cv::Mat im;
    convertToMat(mat,im,rows,CV_32F,true);
    if (flip) {
        cv::transpose(im,im);
        cv::flip(im,im,0);
    }
    cv::imshow(title, im); cv::waitKey(waitKey);
}
