#include "charsDetection.h"


using namespace std;
using namespace cv;

void detectChars(Mat& image, Mat& result) {

	Mat src = image.clone();
	Mat thresh;

	//Perform some preprocessing______________________________________________________________________
	cvtColor(src, src, COLOR_BGR2GRAY); /// Go Gray

	threshold(src, thresh, 127, 255, THRESH_BINARY_INV + THRESH_OTSU); ///Perform some thresholding

	double dec = 0.50;
	for (int k = 0; k < 3; k++) {		
		
		resize(thresh, thresh, Size(), 1 + dec, 1 + dec);
		threshold(thresh, thresh, 127, 255, THRESH_BINARY + THRESH_OTSU);///Perform some thresholding
		dec /= 2;
	}
	/*
	resize(thresh, thresh, Size(), 1.25, 1.25);
	threshold(thresh, thresh, 127, 255, THRESH_BINARY + THRESH_OTSU);///Perform some thresholding

	resize(thresh, thresh, Size(), 1.1, 1.1);
	threshold(thresh, thresh, 127, 255, THRESH_BINARY + THRESH_OTSU);///Perform some thresholding
	*/
	Canny(thresh, thresh, 100, 200);///Perform Canny algorithm
	//________________________________________________________________________________________________

	/// FInd contours of the image
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons and get bounding rects________________________________________
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());	

	for (int i = 0; i < contours.size(); i++){

		approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true);
		boundRect[i] = boundingRect(Mat(contours[i]));
		
	}
	///________________________________________________________________________________________________



	///________________________________________________________________________________________________
	int hgt = thresh.rows;
	int wid = thresh.cols;

	/// Draw bonding rects
	//Mat drawing = Mat::zeros(thresh.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++){

		bool min_dims = (boundRect[i].width > wid / 75) && (boundRect[i].height > hgt / 10);
		bool max_dims = (boundRect[i].width < wid / 2) && (boundRect[i].height < hgt );
		Scalar color = (0, 255, 255);

		//drawContours(thresh, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		if ( min_dims && max_dims ) {

			rectangle(thresh, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		}
	}
	///__________________________________________________________________________________________________

	result = thresh.clone();


}

