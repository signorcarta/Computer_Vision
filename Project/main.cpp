#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#include "PreProcessing.h"
#include "PlateDetection.h"
#include "PlatesExtraction.h"
#include "charsDetection.h"


//#include "classifier/haarcascade_russian_plate_number.xml"



//#define SHOW_STEPS /// Comment/uncomment this line to hide/show intermediate steps 
//#define DISABLE /// 

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	//Loading image________________________________________________________________________________

	Mat src = imread("C:\\Users\\david\\source\\repos\\License_plate_recognition\\6.jpg"); /// Source image
	
#ifdef SHOW_STEPS
	cout << "---> Showing original image. \n\n" << endl;
	namedWindow("ORIGINAL IMAGE");
	imshow("ORIGINAL IMAGE", src);
	waitKey();
#endif //SHOW_STEPS

	//____________________________________________________________________________________________
	


	//Plate detection using cascade classifier_____________________________________________________

	Mat detected; /// Image with the drawn rectangles
	vector<Rect> rects; /// Vector of rectangles representing detected plates
	int n_plates = 0; /// Number of detected plates [LAST TIME IS NEEDED]
	String path = "C:\\Users\\david\\source\\repos\\License_plate_recognition\\classifier\\haarcascade_license_plate.xml";
	
	detectPlate(src, detected, path, n_plates, rects);

#ifdef SHOW_STEPS
	cout << "\n---> Showing detected plate.\n";
	namedWindow("DETECTED PLATE");
	imshow("DETECTED PLATE", detected);
	waitKey();
#endif //SHOW_STEPS

	//_____________________________________________________________________________________________




	//Plate extraction_____________________________________________________________________________

	Mat vecOfPlates; /// Vector containing plate cropped from the image

	extractPlate(detected, rects, vecOfPlates);

#ifdef SHOW_STEPS
	showPlate(vecOfPlates);
	cout << "\n---> Showing cropped plate.\n";
#endif //SHOW_STEPS

	//_____________________________________________________________________________________________
	


	/*Plate thresholding___________________________________________________________________________
	
	Mat thresholded_plate; ///Thresholded plate
	
	Preprocess(vecOfPlates, thresholded_plate);

	#ifdef SHOW_STEPS
		cout << "\n---> Showing thresholded plate. \n" << endl;
		namedWindow("THRESHOLDED PLATE");
		imshow("THRESHOLDED PLATE", thresholded_plate);
		waitKey();
	#endif //SHOW_STEPS
	
	
	//_____________________________________________________________________________________________*/

	

	//Chars detection______________________________________________________________________________
		Mat detectedChars;
		Canny(vecOfPlates, detectedChars, 100, 200);
		detectChars(detectedChars, detectedChars);

//#ifdef SHOW_STEPS
		cout << "---> Showing detected chars. \n" << endl;
		namedWindow("DETECTED CHARS");
		imshow("DETECTED CHARS", detectedChars);
		waitKey();
//#endif //SHOW_STEPS

	//_____________________________________________________________________________________________


	return 0;
}
