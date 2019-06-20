#include "charsDetection.h"


using namespace std;
using namespace cv;


void detectChars(Mat& image, Mat& result, vector<Rect>& charsRects) {

	Mat src = image.clone();
	Mat src_ = image.clone();
	Mat filtered;
	Mat thresh;
	Mat canny;



	/// Perform some preprocessing_________________________________________________________________

	cvtColor(src, src, COLOR_BGR2GRAY); /// Go Gray
	bilateralFilter(src, filtered, 9, 100, 100); /// Perform some filtering
	threshold(filtered, thresh, 220, 255, THRESH_BINARY + THRESH_OTSU); /// Perform some thresholding
	Canny(thresh, canny, 100, 200); /// Perform Canny algorithm

	///____________________________________________________________________________________________



	/// FInd contours of the image_________________________________________________________________

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	///____________________________________________________________________________________________



	/// Approximate contours to polygons and get bounding rects____________________________________

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++) {

		approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true);
		boundRect[i] = boundingRect(Mat(contours[i]));

	}

	///____________________________________________________________________________________________



	///Draw bonding rectangles_____________________________________________________________________

	for (int i = 0; i < contours.size(); i++) {

		bool min_dims = boundRect[i].width > 5 && boundRect[i].height > 40;
		bool max_dims = boundRect[i].width < 80 && boundRect[i].height < 100;

		//cout << "\n bounding width: " << boundRect[i].width << endl;
		//cout << "bounding height: " << boundRect[i].height << endl;

		if (min_dims && max_dims) {

			rectangle(src_, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 1, 8, 0);
			charsRects.push_back(boundRect[i]);

		}
		
	}

	result = src_.clone();
	
	///____________________________________________________________________________________________
	
}
