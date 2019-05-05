#ifndef READIMAGE_H
#define READIMAGE_H

#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;

float** Mat2pChar(const Mat& img);
Mat pChar2Mat(unsigned char** pImg, int width, int height, int band);

#endif