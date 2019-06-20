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



/*
	This function detects the plate in the original image and fills a vector of rectangles representing
	its position in the image, then uses it to draw the rectangle in the image
*/
void detectPlate(cv::Mat& image, cv::Mat& detected, cv::String& path, int& platesFound, std::vector<cv::Rect>& plates);

#endif
