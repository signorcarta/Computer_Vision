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
	cv::Mat src_2;
	cv::Mat edit_src;
	cv::Mat result;
	cv::Mat edit_result;	
	cv::Mat b_hist, g_hist, r_hist; /// bgr color space
	cv::Mat l_hist, a_hist, bb_hist; /// Lab color space 
	cv::Mat eq_b_hist, eq_g_hist, eq_r_hist;
	cv::Mat eq_l_hist, eq_a_hist, eq_bb_hist;
	cv::Mat eq_b_plane, eq_g_plane, eq_r_plane;
	cv::Mat eq_l_plane, eq_a_plane, eq_bb_plane;

	std::vector<cv::Mat> bgr_planes;
	std::vector<cv::Mat> eq_bgr_planes;
	std::vector<cv::Mat> lab_planes;

	std::vector<cv::Mat> histogram;
	std::vector<cv::Mat> eq_histogram;

	std::vector<cv::Mat> lab_histogram;	
	std::vector<cv::Mat> eq_lab_histogram;
	std::vector<cv::Mat> dst;
	std::vector<cv::Mat> edit_dst;

	int histSize = 256;
	bool uniform = true;
	bool accumulate = false;



	//Loading and displaying the image__________________________________________________________________
	std::cout << "Loading original image...\n" << std::endl;

	src = cv::imread("C:\\Users\\david\\source\\repos\\Histogram_equalization\\laurea.jpg", 1);
	cv::namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", src);
	waitKey(0);

	std::cout << "Done.\n" << std::endl;
	//__________________________________________________________________________________________________

	src_2 = src.clone(); /// To be used when working on lab color space later

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

	/// Function call
	std::cout << "Showing histograms for \"RGB\" color space: \n";
	showHistogram(histogram);
	std::cout << "Done.\n" << std::endl;

	//__________________________________________________________________________________________________



	//Equalizing each channel___________________________________________________________________________

	cv::equalizeHist(bgr_planes[0], eq_b_plane);
	cv::equalizeHist(bgr_planes[1], eq_g_plane);
	cv::equalizeHist(bgr_planes[2], eq_r_plane);

	/// Compute the new histograms
	cv::calcHist(&eq_b_plane, 1, 0, Mat(), eq_b_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&eq_g_plane, 1, 0, Mat(), eq_g_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&eq_r_plane, 1, 0, Mat(), eq_r_hist, 1, &histSize, &histRange, uniform, accumulate);	

	/// Assembling the equalized image
	dst.push_back(eq_b_plane);
	dst.push_back(eq_g_plane);
	dst.push_back(eq_r_plane);

	merge(dst, result);

	/// Display equalized histogram
	eq_histogram.push_back(eq_b_hist);
	eq_histogram.push_back(eq_g_hist);
	eq_histogram.push_back(eq_r_hist);

	std::cout << "Showing equalized histograms for \"RGB\" color space: \n";
	showHistogram(eq_histogram);

	/// Display equalized image	
	std::cout << "Equalized image: \n";
	cv::namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", result);
	waitKey(0);
	std::cout << "Done.\n" << std::endl;

	//__________________________________________________________________________________________________



	//Working on a different color space________________________________________________________________

	std::cout << "Now working on \"lab\" color space. \n" << std::endl;

	///Calculating histograms___________________________________________________________________________

	/// Transforming the image in a different color space
	cv::cvtColor(src_2, edit_src, COLOR_BGR2Lab);
	
	///Displaying the original image in the new colorspace
	std::cout << "Image in \"lab\" color space: \n";
	cv::namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", edit_src);
	waitKey(0);
	std::cout << "Done.\n" << std::endl;

	/// Separate the image in 3 planes ( L, a and b )	
	cv::split(edit_src, lab_planes);

	/// Compute the three histograms of the new color channels
	cv::calcHist(&lab_planes[0], 1, 0, Mat(), l_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&lab_planes[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&lab_planes[2], 1, 0, Mat(), bb_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Properly filling the vector
	lab_histogram.push_back(l_hist);
	lab_histogram.push_back(a_hist);
	lab_histogram.push_back(bb_hist);

	std::cout << "Showing histograms for \"lab\" color space: \n";
	showHistogram(lab_histogram);
	std::cout << "Done.\n" << std::endl;
	///_________________________________________________________________________________________________

	///Equalizing only one channel______________________________________________________________________

	cv::equalizeHist(lab_planes[0], eq_l_plane); //plane [0] equalization
	eq_a_plane = lab_planes[1].clone();
	eq_bb_plane = lab_planes[2].clone();

	/// Compute the new histogram
	cv::calcHist(&eq_l_plane, 1, 0, Mat(), eq_l_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Display equalized histogram
	eq_lab_histogram.push_back(eq_l_hist);
	eq_lab_histogram.push_back(a_hist);
	eq_lab_histogram.push_back(bb_hist);

	std::cout << "Showing equalized histogram(s) for \"lab\" color space: \n";
	showHistogram(eq_lab_histogram);
	std::cout << "Done.\n" << std::endl;

	/// Assembling the equalized image
	edit_dst.push_back(eq_l_plane);
	edit_dst.push_back(eq_a_plane);
	edit_dst.push_back(eq_bb_plane);

	merge(edit_dst, edit_result);

	/// Display equalized image	
	std::cout << "Image in \"lab\" color with only one equalized channel: \n";
	cv::namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", edit_result);
	waitKey(0);
	std::cout << "Done.\n" << std::endl;

	///_________________________________________________________________________________________________


	//__________________________________________________________________________________________________

	return 0;
}



