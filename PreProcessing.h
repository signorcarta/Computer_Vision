#ifndef PREPROCESS_H
#define PREPROCESS_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;

void Preprocess(Mat &srcImage, Mat &tresholdImage);

Mat maxContrast(Mat grayImage);

#endif
