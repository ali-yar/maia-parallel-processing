#include <iostream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

# define PI 3.14159265358979323846


int main() {
    std::string file = "C:/Users/hp4540/Desktop/sinogram.jpg";
    cv::Mat sinogram = cv::imread(file, CV_LOAD_IMAGE_UNCHANGED);
    // cv::normalize(sinogram,sinogram,0,1,CV_MINMAX,CV_32F);

    cv::imshow("sinogram", sinogram); cv::waitKey(0);

    int numOfParallelProjections = sinogram.rows;
    int numOfAngularProjections  = sinogram.cols;

    int step = 180 / numOfAngularProjections;

    float thetas[90];
    for (int i=0; i<numOfAngularProjections; i++) {
        thetas[i] = (i * step) * (PI/180.);

    }

    cv::Mat im(numOfParallelProjections, numOfParallelProjections, CV_32F, cv::Scalar(0));

    // find the middle index of the projections
    int midindex = floor(numOfParallelProjections/2);

    for (int r=0; r<im.rows; r++) {
        for (int c=0; c<im.cols; c++) {
            for (int t=0; t<numOfAngularProjections; t++) {
                float d = thetas[t];
                int x = c - midindex;
                int y = r - midindex;
                int rotCoords = round(midindex + x*sin(d) + y*cos(d));

                if ((rotCoords > -1) && (rotCoords < numOfParallelProjections)) {
                    im.at<float>(r,c) = im.at<float>(r,c) + (sinogram.at<uchar>(rotCoords,t)/numOfAngularProjections);
                }
            }
        }
    }
    cv::normalize(im,im,0,255,CV_MINMAX,CV_8U);
    cv::imshow("im", im); cv::waitKey(0);

    return 0;
}
