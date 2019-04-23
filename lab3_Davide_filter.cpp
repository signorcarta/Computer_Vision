#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

using namespace cv;

// constructor
Filter::Filter(cv::Mat input_img, int size) {

	input_image = input_img;
	if (size % 2 == 0)
		size++;
	filter_size = size;
}


void Filter::doFilter() {
	
	result_image = input_image.clone();

}

// get output of the filter
cv::Mat Filter::getResult() {

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


//Gaussian___________________________________________________________________________________________________
GaussianFilter::GaussianFilter(cv::Mat input_img, int size, double sigma) : Filter(input_img, size) {

	sigma = sigma;
	type = "Gaussian";
}
	
void GaussianFilter::doFilter() {

	result_image = input_image.clone();
	cv::GaussianBlur(input_image, result_image, cv::Size(filter_size, filter_size), sigma, 0, cv::BORDER_DEFAULT);
	
}

void GaussianFilter::setSigma(float value) {

	sigma = value;

}

double GaussianFilter::getSigma() {

	return sigma;

}

//___________________________________________________________________________________________________________


//Median_____________________________________________________________________________________________________
MedianFilter::MedianFilter(cv::Mat input_img, int filter_size) : Filter(input_img, filter_size) {

	type = "Median";

}

void MedianFilter::doFilter() {

	result_image = input_image.clone();
	cv::medianBlur(input_image, result_image, filter_size);

}

//___________________________________________________________________________________________________________


//Bilateral__________________________________________________________________________________________________
BilateralFilter::BilateralFilter(cv::Mat input_img, int filter_size, double sigmaColor, double sigmaSpace) : Filter(input_img, filter_size) {

	sigmaColor = sigma_color;
	sigmaSpace = sigmaSpace;
	type = "Bilateral";

}

void BilateralFilter::doFilter() {

	result_image = input_image.clone();
	cv::bilateralFilter(input_image, result_image, 7, sigma_color, sigma_space, cv::BORDER_DEFAULT);

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

//___________________________________________________________________________________________________________
		

