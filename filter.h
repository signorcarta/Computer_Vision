#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Generic class implementing a filter with the input and output image data and the parameters
class Filter {

	// Methods

public:

	// Constructor 
	/// input_img: image to be filtered
	/// filter_size : size of the kernel/window of the filter
	Filter(cv::Mat input_img, int filter_size);

	/// perform filtering (in base class do nothing, to be reimplemented in the derived filters)
	void doFilter();

	/// get the output of the filter
	cv::Mat getResult();

	/// set the window size (square window of dimensions size x size)
	void setSize(int size);

	/// get the Window Size
	int getSize();

	/// Data

protected:

	/// input image
	cv::Mat input_image;

	/// output image (filter result)
	cv::Mat result_image;

	/// window size
	int filter_size;



};

// Medianian Filter________________________________________________________________________________
class MedianFilter : public Filter {

	/// place constructor
	public : 
		MedianFilter(cv::Mat src, int ksize, cv::Mat dst) :
		Filter(input_image, filter_size) {
		dst = result_image;
		}

		/// re-implement  doFilter()
		void doFilter() {
			cv::medianBlur(input_image, result_image, filter_size);
	
		}

		// no additional parameters
};
//_________________________________________________________________________________________________


//Gaussian Filter__________________________________________________________________________________
class GaussianFilter : public Filter {

	/// place constructor
	public:
		GaussianFilter(cv::Mat src, int ksize, cv::Mat dst, double sigma_X, double sigma_Y) :
			Filter(input_image, filter_size) {
			dst = result_image;
			sigma_X = sigmaX;
			sigma_Y = sigmaY;
		}


		/// re-implement  doFilter()
		void doFilter() {

			cv::GaussianBlur(input_image, result_image, size, sigmaX, sigmaY);

		}

		/// additional parameter: standard deviation (sigma)
		double sigmaX;
		double sigmaY;
		cv::Size size;

};
//_________________________________________________________________________________________________


//Bilateral filter_________________________________________________________________________________
class BilateralFilter : public Filter {

	/// place constructor
	public:
		BilateralFilter (cv::Mat src, int ksize, cv::Mat dst, double sigma_range, double sigma_space) :
			Filter(input_image, filter_size) {
			dst = result_image;
			sigma_range = sigmaRange;
			sigma_space = sigmaSpace;
		}

	/// re-implement  doFilter()
	void doFilter() {

		cv::bilateralFilter(input_image, result_image, filter_size, sigmaRange, sigmaSpace);

	}
	
	/// additional parameters: sigma_range, sigma_space
	double sigmaRange;
	double sigmaSpace;

};

