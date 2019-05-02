#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

using namespace std;
using namespace cv;

// constructor
Filter::Filter(Mat input_img, int size) {

	input_image = input_img;

	if (size % 2 == 0)
		size++;
	filter_size = size;
}


void Filter::doFilter() {
	
	result_image = input_image.clone();

}

// get output of the filter
Mat Filter::getResult() {

	return result_image;
}

//set window size (it needs to be odd)
void Filter::setSize(int size) {

	if (size % 2 == 0)
		size++;
	filter_size = size;
}

//get window size 
int Filter::getSize() {

	return filter_size;
}


///Gaussian___________________________________________________________________________________________________
GaussianFilter::GaussianFilter(Mat input_img, int size, double Sigma) : Filter(input_img, size) {

	sigma = Sigma;

}
	
void GaussianFilter::doFilter() {

	result_image = input_image.clone();
	GaussianBlur(input_image, result_image, Size(filter_size, filter_size), sigma, 0, BORDER_DEFAULT);
	
}

void GaussianFilter::setSigma(float value) {

	sigma = value;

}

double GaussianFilter::getSigma() {

	return sigma;

}

///___________________________________________________________________________________________________________


///Median_____________________________________________________________________________________________________
MedianFilter::MedianFilter(Mat input_img, int filter_size) : Filter(input_img, filter_size) {}

void MedianFilter::doFilter() {

	result_image = input_image.clone();
	medianBlur(input_image, result_image, filter_size);

}

///___________________________________________________________________________________________________________


///Bilateral__________________________________________________________________________________________________
BilateralFilter::BilateralFilter(cv::Mat input_img, int filter_size, double sigmaColor, double sigmaSpace) : Filter(input_img, filter_size) {

	sigma_color = sigmaColor;
	sigma_space =  sigmaSpace;

}

void BilateralFilter::doFilter() {

	result_image = input_image.clone();
	bilateralFilter(input_image, result_image, 7, sigma_color, sigma_space, cv::BORDER_DEFAULT);

}

void BilateralFilter::setSigmaColor(float value) {

	sigma_color = value;

}

void BilateralFilter::setSigmaSpace(float value) {

	sigma_space = value;

}

double BilateralFilter::getSigmaColor() {

	return sigma_color;
}

double BilateralFilter::getSigmaSpace() {

	return sigma_space;
}

///___________________________________________________________________________________________________________
		

