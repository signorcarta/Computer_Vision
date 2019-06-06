#include "charsDetection.h"


using namespace std;
using namespace cv;

void detectChars(Mat& image, Mat& result) {
	
	Mat src_plate = image.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	String outText;
	
	int n_contours;

	findContours(src_plate, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	n_contours = contours.size();

	
	for (size_t i = 0; i < contours.size(); i++){
		
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(src_plate, contours, (int)i, color, 1, 8, hierarchy, 0, Point());
	}
	
}

