#ifndef PLATE_EXTRACTION_H
#define PLATE_EXTRACTION_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/*
	This function gets as input the image and the vector of rectangles of the detected plates
	and crops the original image filling a vector with the crops of the original image that
	should be the detected plates
*/
void extractPlate(Mat& image, vector<Rect>& rects, vector<Mat>& plates);

/*
	This function gets in iput the vector of images of the detected plates and displays them
*/
void showPlate(vector<Mat>& plates);

#endif
