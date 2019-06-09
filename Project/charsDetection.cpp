#include "charsDetection.h"


using namespace std;
using namespace cv;

void detectChars(Mat& image, Mat& result, vector<Rect>& charsRects) {

	Mat src = image.clone();
	Mat thresh;
	Mat canny;

	/// Perform some preprocessing_________________________________________________________________
	cvtColor(src, src, COLOR_BGR2GRAY); /// Go Gray
	threshold(src, thresh, 127, 255, THRESH_BINARY_INV + THRESH_OTSU); ///Perform some thresholding

	resize(thresh, thresh, Size(), 1.75, 1.75);
	threshold(thresh, thresh, 127, 255, THRESH_BINARY + THRESH_OTSU); ///Perform some thresholding
		
	resize(thresh, thresh, Size(), 1.5, 1.5);
	threshold(thresh, thresh, 127, 255, THRESH_BINARY_INV + THRESH_OTSU); ///Perform some thresholding

	resize(thresh, thresh, Size(), 1.25, 1.25);
	threshold(thresh, thresh, 127, 255, THRESH_BINARY + THRESH_OTSU); ///Perform some thresholding
	
	Canny(thresh, canny, 100, 200);///Perform Canny algorithm

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
	int hgt = canny.rows;
	int wid = canny.cols;
	//Mat drawing = zeros(thresh.size(), CV_8UC3);
	
	for (int i = 0; i < contours.size(); i++){

		bool min_dims = (boundRect[i].width > (wid/70) ) && (boundRect[i].height > (hgt / 2) );
		bool max_dims = (boundRect[i].width < (wid/3)  ) && (boundRect[i].height < (hgt)       );
		Scalar color = (0, 255, 255);
		//drawContours(thresh, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());

		if ( min_dims && max_dims ) {
			
			rectangle(canny, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			charsRects.push_back(boundRect[i]);

		}
	}
	///____________________________________________________________________________________________

	result = canny.clone();

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

