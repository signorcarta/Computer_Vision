#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#include "PreProcessing.h"
#include "PlateDetection.h"
#include "PlatesExtraction.h"



#define SHOW_STEPS /// Comment/uncomment this line to hide/show intermediate steps 
//#define PREPROCESS /// If preprocess is disabled ==> change input image in detectPlate()

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	//Loading image
	Mat src = imread("C:\\Users\\david\\source\\repos\\License_plate_recognition\\7.jpg");
	cout << "Showing original image. \n\n" << endl;
	namedWindow("ORIGINAL IMAGE");
	imshow("ORIGINAL IMAGE", src);
	waitKey();

	//Preprocessing
#ifdef PREPROCESS
	Mat tresholded;
	Preprocess(src, tresholded);
#ifdef SHOW_STEPS
	cout << "Showing tresholded image. \n\n" << endl;
	namedWindow("TRESHOLDED IMAGE");
	imshow("TRESHOLDED IMAGE", tresholded);
	waitKey();
#endif
#endif

	//Plate detection using cascade classifier
	Mat detected;
	vector<Rect> rects;
	string path = "C:\\Users\\david\\source\\repos\\License_plate_recognition\\classifier\\haarcascade_russian_plate_number.xml";
	int n_plates = 0;

	detectPlate(src, detected, path, n_plates, rects);

#ifdef SHOW_STEPS
	cout << "Showing possible detected plates.\n";
	cout << "Found " << n_plates << " possible plates.\n\n" << endl;
	namedWindow("DETECTED PLATE");
	imshow("DETECTED PLATE", detected);
	waitKey();
#endif

	//Plate extraction
	vector<Mat> vecOfPlates;
	extractPlate(src, rects, vecOfPlates);

	return 0;
}
