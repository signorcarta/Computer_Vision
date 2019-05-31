#include "PlatesExtraction.h"

using namespace std;
using namespace cv;


void extractPlate(Mat& image, vector<Rect>& rects, vector<Mat>& plates) {

	int howMany = rects.size(); /// How many plates have actually been found

	for (size_t i = 0; i < howMany; i++) {

		Mat temp = image.clone();
		plates[i] = temp(rects[i]); /// FIlling the vector with each detected plate

	}
	
	showPlate(plates);
}


void showPlate(vector<Mat>& plates) {

	Mat result;

	for (size_t i = 0; i<plates.size(); i++) {

		result = plates[i].clone();
		namedWindow("Showing plate found");
		imshow("Showing plate found", result);
		waitKey();
	}
}

/*
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//          NOTE: These functions do not work. 
//                Especially I am not sure about how to crop the image given the rectangles.
//
///////////////////////////////////////////////////////////////////////////////////////////////////
*/
