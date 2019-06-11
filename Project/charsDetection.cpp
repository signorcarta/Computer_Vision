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
	threshold(filtered, thresh, 127, 255, THRESH_BINARY + THRESH_OTSU);
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

	for (int i = 0; i < contours.size(); i++){

		approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true);
		boundRect[i] = boundingRect(Mat(contours[i]));		
				
	}
	///____________________________________________________________________________________________



	///Draw bonding rectangles_____________________________________________________________________
		
	for (int i = 0; i < contours.size(); i++){
		Scalar color = (0, 255, 255);
		bool min_dims = boundRect[i].width > 10 && boundRect[i].height > 40;
		bool max_dims = boundRect[i].width < 80 && boundRect[i].height < 100;
			
		if ( min_dims && max_dims) {
			
			rectangle(src_, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			charsRects.push_back(boundRect[i]);			

		}
	}
	///____________________________________________________________________________________________

	result = src_.clone();

}

void extractChars(Mat& image, vector<Rect>& charsRects, vector<Mat>& charsCollection) {

	Mat image_ = image.clone();
	vector<Rect> charsRects_ ; 

	///Get a copy of the array of rects
	for (int i=0; i<charsRects.size(); i++) {

		charsRects_.push_back(charsRects[i]);
	}



	/// Sort the array of rectangles__________________________
	struct byPosition {

		bool operator() (Rect& a, const Rect& b) {
			return a.x < b.x;
		}

	};
	  
	sort(charsRects_.begin(), charsRects_.end(), byPosition());
	///_______________________________________________________


	/// FIll the output vector with ordered chars_____________
	for (int i = 0; i < charsRects_.size(); i++) {

		Mat original = image_.clone();
		charsCollection.push_back(original(charsRects_[i]));
		
	}
	///_______________________________________________________
}

void showChars(vector<Mat>& charsCollection) {

	for (int i = 0; i < charsCollection.size(); i++) {

		Mat temp = charsCollection[i].clone();

		namedWindow("Char");
		imshow("Char", temp);
		waitKey();

	}

}

