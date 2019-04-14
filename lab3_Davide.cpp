#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	Mat src, dst;
	Mat b_hist, g_hist, r_hist;
	Mat eq_b_hist, eq_g_hist, eq_r_hist;
	vector<Mat> bgr_planes;
	
	int histSize = 256;
	bool uniform = true; 
	bool accumulate = false;

	//Loading and displaying the image_______________________________________________________________
	src = cv::imread("C:\\Users\\david\\source\\repos\\Histogram_equalization\\lena.png");
	cv::namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", src);
	waitKey(0);
	//_______________________________________________________________________________________________

	//Calculating histograms_________________________________________________________________________

	/// Separate the image in 3 planes ( B, G and R )	
	split(src, bgr_planes);

	/// Set the ranges (for B,G,R))
	float range[] = { 0, 256 };
	const float* histRange = { range };	

	/// Compute the three histograms
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat eq_histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	
	/*
	Normalize the result to [ 0, histImage.rows ], so that its values fall 
	in the range indicated by the parameters entered.
	*/
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	namedWindow("calcHist Demo", WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);
	waitKey(0);
	//_______________________________________________________________________________________________

	//Equalizing each channel________________________________________________________________________
	equalizeHist(src, dst);

	/// Separate the image in 3 planes ( B, G and R )	
	split(dst, bgr_planes);
		
	/// Compute the three histograms
	calcHist(&bgr_planes[0], 1, 0, Mat(), eq_b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), eq_g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), eq_r_hist, 1, &histSize, &histRange, uniform, accumulate);

	normalize(eq_b_hist, eq_b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(eq_g_hist, eq_g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(eq_r_hist, eq_r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	
	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(eq_b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(eq_b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(eq_g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(eq_g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(eq_r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	// Display histogrms of equalized channels and images
	namedWindow("calcHist Demo", WINDOW_AUTOSIZE);
	imshow("calcHist Demo", eq_histImage);
	waitKey(0);
	//_______________________________________________________________________________________________

	return 0;
}
