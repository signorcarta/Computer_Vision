#include "PlatesExtraction.h"

using namespace std;
using namespace cv;


void extractPlate(Mat& image, vector<Rect>& rects, vector<Mat>& plates, int& n_of_plates) {

	Mat temp;

	for (size_t i = 0; i < n_of_plates; i++) {

		temp = image.clone();
		plates.push_back(temp(rects[i])); /// FIlling the vector with each detected plate
		
	}
		
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
