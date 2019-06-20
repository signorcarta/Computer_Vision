#include "PlatesExtraction.h"

using namespace std;
using namespace cv;



void extractPlate(Mat& image, vector<Rect>& rects, Mat& plate) {

	Mat temp;
	Mat plate_;
	
	temp = image.clone();
	plate_ = temp(rects[0]); /// FIlling the vector with detected plate
	resize(plate_, plate_, Size(300, 100));
	plate = plate_.clone();
			
}


void showPlate(Mat& plate) {

	Mat result;

	result = plate.clone();
	namedWindow("plate");
	imshow("plate", result);
	waitKey();
	
}
