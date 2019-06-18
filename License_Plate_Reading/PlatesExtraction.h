#ifndef PLATE_EXTRACTION_H
#define PLATE_EXTRACTION_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>



/*
	This function gets as input the image and the vector of rectangles of the detected plates
	and crops the original image filling a vector with the crops of the original image that
	should be the detected plates
*/
void extractPlate(cv::Mat& image, std::vector<cv::Rect>& rects, cv::Mat& plate);

/*
	This function gets as iput the vector of images of the detected plates and displays them
*/
void showPlate(cv::Mat& plates);

#endif
