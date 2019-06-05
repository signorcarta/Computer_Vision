#include "PlatesExtraction.h"

using namespace std;
using namespace cv;


void extractPlate(Mat& image, vector<Rect>& rects, Mat& plate) {

	Mat temp;
	
	temp = image.clone();
	plate = temp(rects[0]); /// FIlling the vector with detected plate
			
}


void showPlate(Mat& plate) {

	Mat result;

	result = plate.clone();
	namedWindow("plate");
	imshow("plate", result);
	waitKey();
	
}
