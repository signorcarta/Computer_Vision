#ifndef PREPROCESS_H
#define PREPROCESS_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;

void Preprocess(Mat &srcImage, Mat &processed);

Mat maxContrast(Mat grayImage);

void getHistograms(Mat& src, Mat& ver, Mat& hor);

void showHistograms(Mat& ver, Mat& hor);

#endif
