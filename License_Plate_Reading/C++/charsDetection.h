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


/*
	This function detects the chars in the plate and draw a green rectangle around each one
*/
void detectChars(cv::Mat& image, cv::Mat& result, std::vector<cv::Rect>& charsRects);
	
#endif
