#ifndef CHARS_EXTRACTION_H
#define CHARS_EXTRACTION_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>



/*
	This function extract the chars from the plate and fills a vector containing them
*/
void extractChars(cv::Mat& image, std::vector<cv::Rect>& charsRects, std::vector<cv::Mat>& charsCollection_resized);

/*
	This function gets as input the vector of cropped chars and shows them
*/
void showChars(std::vector<cv::Mat>& charsCollection_resized);

#endif
