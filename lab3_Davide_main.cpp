#include <math.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d.hpp>
#include "filter.h"

using namespace std;
using namespace cv;

/// Function provided that shows the histograms________________________________________________________
void showHistogram(vector<Mat>& hists)
{
	// Min/Max computation
	double hmax[3] = { 0,0,0 };
	double min;
	minMaxLoc(hists[0], &min, &hmax[0]);
	minMaxLoc(hists[1], &min, &hmax[1]);
	minMaxLoc(hists[2], &min, &hmax[2]);

	string wname[3] = { "blue", "green", "red" };
	Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

	vector<Mat> canvas(hists.size());

	// Display each histogram in a canvas
	for (int i = 0, end = hists.size(); i < end; i++)
	{
		canvas[i] = Mat::ones(125, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
		{
			cv::line(
				canvas[i],
				Point(j, rows),
				Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
				hists.size() == 1 ? Scalar(200, 200, 200) : colors[i],
				1, 8, 0
			);
		}

		imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
		waitKey(0);
	}
}
///____________________________________________________________________________________________________



int main(int argc, char** argv){

	//Parameters inizialization
	Mat src;
	Mat src_2;
	Mat edit_src;
	Mat result;
	Mat edit_result;
	Mat b_hist, g_hist, r_hist; /// bgr color space
	Mat l_hist, a_hist, bb_hist; /// Lab color space 
	Mat eq_b_hist, eq_g_hist, eq_r_hist;
	Mat eq_l_hist, eq_a_hist, eq_bb_hist;
	Mat eq_b_plane, eq_g_plane, eq_r_plane;
	Mat eq_l_plane, eq_a_plane, eq_bb_plane;

	vector<Mat> bgr_planes;
	vector<Mat> eq_bgr_planes;
	vector<Mat> lab_planes;

	vector<Mat> histogram;
	vector<Mat> eq_histogram;

	vector<Mat> lab_histogram;
	vector<Mat> eq_lab_histogram;
	vector<Mat> dst;
	vector<Mat> edit_dst;

	int histSize = 256;
	bool uniform = true;
	bool accumulate = false;

	

	//Loading and displaying the image__________________________________________________________________
	cout << "Loading original image...\n" << endl;

	src = imread("C:\\Users\\david\\source\\repos\\Histogram_equalization\\image.jpg", 1);
	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", src);
	waitKey(0);

	std::cout << "Done.\n\n" << std::endl;
	//__________________________________________________________________________________________________

	src_2 = src.clone(); /// To be used when working on lab color space later

	//Calculating histograms____________________________________________________________________________

	/// Separate the image in 3 planes ( B, G and R )	
	split(src, bgr_planes);

	/// Set the ranges (for B,G,R))
	float range[] = { 0, 256 };
	const float* histRange = { range };

	/// Compute the three histograms
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Properly filling the vector to be passed as parameter in the function below
	histogram.push_back(b_hist);
	histogram.push_back(g_hist);
	histogram.push_back(r_hist);

	/// Function call
	cout << "Showing histograms for \"RGB\" color space: \n";
	showHistogram(histogram);
	cout << "Done.\n\n" << endl;

	//__________________________________________________________________________________________________



	//Equalizing each channel___________________________________________________________________________

	equalizeHist(bgr_planes[0], eq_b_plane);
	equalizeHist(bgr_planes[1], eq_g_plane);
	equalizeHist(bgr_planes[2], eq_r_plane);

	/// Compute the new histograms
	calcHist(&eq_b_plane, 1, 0, Mat(), eq_b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&eq_g_plane, 1, 0, Mat(), eq_g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&eq_r_plane, 1, 0, Mat(), eq_r_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Assembling the equalized image
	dst.push_back(eq_b_plane);
	dst.push_back(eq_g_plane);
	dst.push_back(eq_r_plane);

	merge(dst, result);

	/// Display equalized histogram
	eq_histogram.push_back(eq_b_hist);
	eq_histogram.push_back(eq_g_hist);
	eq_histogram.push_back(eq_r_hist);

	cout << "Showing equalized histograms for \"RGB\" color space: \n";
	showHistogram(eq_histogram);
	cout << "Done.\n" << endl;

	/// Display equalized image	
	cout << "Equalized image: \n\n";
	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", result);
	waitKey(0);

	//__________________________________________________________________________________________________



	//Working on a different color space________________________________________________________________

	cout << "Now working on \"lab\" color space. \n" << endl;

	///Calculating histograms___________________________________________________________________________

	/// Transforming the image in a different color space
	cvtColor(src_2, edit_src, COLOR_BGR2Lab);

	///Displaying the original image in the new colorspace
	cout << "Image in \"lab\" color space: \n\n";
	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", edit_src);
	waitKey(0);

	/// Separate the image in 3 planes ( L, a and b )	
	cv::split(edit_src, lab_planes);

	/// Compute the three histograms of the new color channels
	calcHist(&lab_planes[0], 1, 0, Mat(), l_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&lab_planes[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&lab_planes[2], 1, 0, Mat(), bb_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Properly filling the vector
	lab_histogram.push_back(l_hist);
	lab_histogram.push_back(a_hist);
	lab_histogram.push_back(bb_hist);

	cout << "Showing histograms for \"lab\" color space: \n";
	showHistogram(lab_histogram);
	cout << "Done.\n\n" << endl;
	///_________________________________________________________________________________________________

	///Equalizing only one channel______________________________________________________________________

	equalizeHist(lab_planes[0], eq_l_plane); //plane [0] equalization
	eq_a_plane = lab_planes[1].clone();
	eq_bb_plane = lab_planes[2].clone();

	/// Compute the new histogram
	calcHist(&eq_l_plane, 1, 0, Mat(), eq_l_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Display equalized histogram
	eq_lab_histogram.push_back(eq_l_hist);
	eq_lab_histogram.push_back(a_hist);
	eq_lab_histogram.push_back(bb_hist);

	cout << "Showing equalized histogram(s) for \"lab\" color space: \n";
	showHistogram(eq_lab_histogram);
	cout << "Done.\n\n" << endl;

	/// Assembling the equalized image
	edit_dst.push_back(eq_l_plane);
	edit_dst.push_back(eq_a_plane);
	edit_dst.push_back(eq_bb_plane);

	merge(edit_dst, edit_result);
	cvtColor(edit_result, edit_result, COLOR_Lab2BGR);

	/// Display equalized image	
	cout << "Image in \"lab\" color space with only one equalized channel: \n";
	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", edit_result);
	waitKey(0);
	cout << "Done.\n\n" << endl;

	///_________________________________________________________________________________________________
	
	//__________________________________________________________________________________________________
	
	

	return 0;
}



