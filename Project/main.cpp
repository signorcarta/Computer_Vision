#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#include "PreProcessing.h"
#include "PlateDetection.h"
#include "PlatesExtraction.h"
//#include "classifier/haarcascade_russian_plate_number.xml"



#define SHOW_STEPS /// Comment/uncomment this line to hide/show intermediate steps 
//#define PREPROCESS /// If preprocess is disabled ==> change input image in detectPlate()

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	//Loading image________________________________________________________________________________

	Mat src = imread("C:\\Users\\david\\source\\repos\\License_plate_recognition\\3.jpg"); /// Source image
	
#ifdef SHOW_STEPS
	cout << "---> Showing original image. \n\n" << endl;
	namedWindow("ORIGINAL IMAGE");
	imshow("ORIGINAL IMAGE", src);
	waitKey();
#endif

	//_____________________________________________________________________________________________



	//Preprocessing________________________________________________________________________________

#ifdef PREPROCESS
	Mat tresholded;

	Preprocess(src, tresholded);

#ifdef SHOW_STEPS
	cout << "---> Showing tresholded image. \n\n" << endl;
	namedWindow("TRESHOLDED IMAGE");
	imshow("TRESHOLDED IMAGE", tresholded);
	waitKey();
#endif
#endif

	//_____________________________________________________________________________________________



	//Plate detection using cascade classifier_____________________________________________________

	Mat detected; /// Image with the drawn rectangles
	vector<Rect> rects; /// Vector of rectangles representing detected plates
	String path = "C:\\Users\\david\\source\\repos\\License_plate_recognition\\classifier\\haarcascade_russian_plate_number.xml";
	int n_plates = 0; /// Number of detected plates

	detectPlate(src, detected, path, n_plates, rects);

#ifdef SHOW_STEPS
	cout << "\n---> Showing possible detected plates.\n";
	if (n_plates<2) {cout << "---> Found just " << n_plates << " plate.\n\n" << endl;}
	else {cout << "---> Found " << n_plates << " possible plates.\n\n" << endl;}	
	namedWindow("DETECTED PLATE");
	imshow("DETECTED PLATE", detected);
	waitKey();
#endif

	//_____________________________________________________________________________________________




	//Plate extraction_____________________________________________________________________________

	vector<Mat> vecOfPlates; /// Vector containing possible plates cropped from the image

	extractPlate(detected, rects, vecOfPlates, n_plates);
#ifdef SHOW_STEPS
	showPlate(vecOfPlates);
#endif

	//_____________________________________________________________________________________________
	

	return 0;
}
