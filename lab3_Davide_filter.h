#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

// Generic class implementing a filter with the input and output image data and the parameters
class Filter {

	// Methods

public:

	// constructor 
	// input_img: image to be filtered
	// filter_size : size of the kernel/window of the filter
	Filter(cv::Mat input_img, int filter_size);

	// perform filtering (in base class do nothing, to be reimplemented in the derived filters)
	void doFilter();

	// get the output of the filter
	cv::Mat getResult();

	//set the window size (square window of dimensions size x size)
	void setSize(int size);

	//get the Window Size
	int getSize();

	String getType();

	// Data

protected:

	// input image
	cv::Mat input_image;

	// output image (filter result)
	cv::Mat result_image;

	// window size
	int filter_size;

	String type;

};



///Gaussian Filter________________________________________________________
class GaussianFilter : public Filter {

	// place constructor
	GaussianFilter(cv::Mat input_img, int filter_size, double sigma);

	void doFilter();

	void setSigma(float value);

	double getSigma();

	// additional parameter: standard deviation (sigma)
protected:

	double sigma;

};
///_______________________________________________________________________


///Median Filter__________________________________________________________
class MedianFilter : public Filter {
	
	// place constructor
	MedianFilter(cv::Mat src, int size);
	
	void doFilter();
	
};
///_______________________________________________________________________


///Bilateral Filter_______________________________________________________
class BilateralFilter : public Filter {

	
public:
	// place constructor
	BilateralFilter(cv::Mat src, int size, double sigmaColor, double sigmaSpace);

	void doFilter();

	void setSigmaColor(float value);

	void setSigmaSpace(float value);

	double getSigmaColor();

	double getSigmaSpace();

protected:

	// additional parameters: sigma_color, sigma_space
	double sigma_color;
	double sigma_space;
	   
};
///_______________________________________________________________________
