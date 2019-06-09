#ifndef CHARS_DETECTION_H
#define CHARS_DETECTION_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>


using namespace cv;
using namespace std;

void detectChars(Mat& image, Mat& result, vector<Rect>& charsRects);

void extractChars(Mat& image, vector<Rect>& charsRects, vector<Mat>& charsCollection);

void showChars(vector<Mat>& charsCollection);

#endif
