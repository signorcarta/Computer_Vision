#ifndef PLATE_DETECTION_H
#define PLATE_DETECTION_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;

void detectPlate(Mat& image, Mat& detected);

#endif