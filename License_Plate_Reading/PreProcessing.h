#ifndef PREPROCESS_H
#define PREPROCESS_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;

void Preprocess(Mat& srcImage, Mat& processed);

Mat maxContrast(Mat grayImage);





#endif