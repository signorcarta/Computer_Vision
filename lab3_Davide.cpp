#include <math.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;



/// Function provided that shows the histograms________________________________________________________
void showHistogram(std::vector<cv::Mat>& hists)
{
	// Min/Max computation
	double hmax[3] = { 0,0,0 };
	double min;
	cv::minMaxLoc(hists[0], &min, &hmax[0]);
	cv::minMaxLoc(hists[1], &min, &hmax[1]);
	cv::minMaxLoc(hists[2], &min, &hmax[2]);

	std::string wname[3] = { "blue", "green", "red" };
	cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
							 cv::Scalar(0,0,255) };

	std::vector<cv::Mat> canvas(hists.size());

	// Display each histogram in a canvas
	for (int i = 0, end = hists.size(); i < end; i++)
	{
		canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
		{
			cv::line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
				hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
				1, 8, 0
			);
		}

		cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
		waitKey(0);
	}
}
///____________________________________________________________________________________________________



int main(int argc, char** argv) {

	//Parameters inizialization
	cv::Mat src;
	cv::Mat result;
	cv::Mat b_hist, g_hist, r_hist;
	cv::Mat eq_b_hist, eq_g_hist, eq_r_hist;
	cv::Mat eq_b_plane, eq_g_plane, eq_r_plane;

	std::vector<cv::Mat> bgr_planes;
	std::vector<cv::Mat> eq_bgr_planes;
	std::vector<cv::Mat> histogram;
	std::vector<cv::Mat> eq_histogram;
	std::vector<cv::Mat> dst;
	
	int histSize = 256;
	bool uniform = true; 
	bool accumulate = false;



	//Loading and displaying the image__________________________________________________________________
	src = cv::imread("C:\\Users\\david\\source\\repos\\Histogram_equalization\\laurea.jpg", 1);
	cv::namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", src);
	waitKey(0);
	//__________________________________________________________________________________________________



	//Calculating histograms____________________________________________________________________________

	/// Separate the image in 3 planes ( B, G and R )	
	cv::split(src, bgr_planes);

	/// Set the ranges (for B,G,R))
	float range[] = { 0, 256 };
	const float* histRange = { range };	

	/// Compute the three histograms
	cv::calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	
	/// Properly filling the vector to be passed as parameter in the function below
	histogram.push_back(b_hist);
	histogram.push_back(g_hist);
	histogram.push_back(r_hist);

	///Function call
	showHistogram(histogram);
		
	//__________________________________________________________________________________________________



	//Equalizing each channel___________________________________________________________________________
	
	cv::equalizeHist(bgr_planes[0], eq_b_plane);
	cv::equalizeHist(bgr_planes[1], eq_g_plane);
	cv::equalizeHist(bgr_planes[2], eq_r_plane);

	cv::calcHist(&eq_b_plane, 1, 0, Mat(), eq_b_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&eq_g_plane, 1, 0, Mat(), eq_g_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&eq_r_plane, 1, 0, Mat(), eq_r_hist, 1, &histSize, &histRange, uniform, accumulate);
		
	eq_histogram.push_back(eq_b_hist);
	eq_histogram.push_back(eq_g_hist);
	eq_histogram.push_back(eq_r_hist);

	dst.push_back(eq_b_plane);
	dst.push_back(eq_g_plane);
	dst.push_back(eq_r_plane);

	///Assembling the equalized image
	merge(dst, result);

	/// Display equalized histogram
	showHistogram(eq_histogram);	

	/// Display equalized image	
	cv::namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", result);
	waitKey(0);
	
	//__________________________________________________________________________________________________
	


	return 0;
}


	
