#ifndef PLATE_DETECTION_H
#define PLATE_DETECTION_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>;

using namespace cv;
using namespace std;

/*
	This function detects possible plates in the originale image and fills a vector of rectangles representing
	their position in the image, then uses them to draw them in the image
*/
void detectPlate(Mat& image, Mat& detected, String& path, int& platesFound, vector<Rect>& plates);

#endif
