#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#include "PreProcessing.h"
#include "PlateDetection.h"
#include "PlatesExtraction.h"
#include "charsDetection.h"
#include "charsExtraction.h"


//#define SHOW_STEPS /// Comment/uncomment this line to hide/show intermediate steps  

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	//Image loading________________________________________________________________________________

	Mat src = imread("images\\1.jpg"); /// Source image
		
#ifdef SHOW_STEPS
	cout << "---> Showing original image. \n\n" << endl;
	namedWindow("ORIGINAL IMAGE");
	imshow("ORIGINAL IMAGE", src);
	waitKey();
#endif //SHOW_STEPS

	//_____________________________________________________________________________________________
	


	//Plate detection _____________________________________________________________________________

	Mat detected; /// Image with the drawn rectangle
	vector<Rect> rects; /// Vector of rectangles representing detected plates
	int n_plates = 0; /// Number of detected plates [LAST TIME IS NEEDED]
	String path = "classifier\\haarcascade_license_plate.xml";
	
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
	cout << "\n---> Showing cropped plate.\n" << endl;
	showPlate(vecOfPlates);
#endif //SHOW_STEPS

	//_____________________________________________________________________________________________
	


	//Chars detection______________________________________________________________________________
	
	Mat detectedChars; /// Image with bounded chars
	vector<Rect> charsRects; /// Vector containing the rectangles related to chars
	
	detectChars(vecOfPlates, detectedChars, charsRects);

#ifdef SHOW_STEPS
	cout << "\n---> Showing detected chars. \n" << endl;
	namedWindow("DETECTED CHARS");
	imshow("DETECTED CHARS", detectedChars);
	waitKey();
#endif //SHOW_STEPS

	//_____________________________________________________________________________________________



	//Chars extraction_____________________________________________________________________________
	vector<Mat> singleChars; /// Vector contatining the cropped chars
	
	extractChars(vecOfPlates, charsRects, singleChars);

//#ifdef SHOW_STEPS	
	if (charsRects.empty()) { cout << "--->   Couldn't read the plate !   <---" << endl; }
	else{ 
		cout << "---> Showing cropped chars.\n";
		showChars(singleChars); 
	}	
//#endif //SHOW_STEPS

	//_____________________________________________________________________________________________



	//Chars saving_________________________________________________________________________________
	if (!charsRects.empty()) {
		for (int i = 0; i < singleChars.size(); i++) {
			/// Every char/letter gets saved twice so:
			if (i % 2 == 0) {

				int j = i / 2; /// Actual index name
				Mat ch = singleChars[i].clone();
				imwrite("chars\\" + to_string(j) + ".jpg", ch);
				cout << "Char {" + to_string(j) + "} saved in folder \"chars\" as " + to_string(j) + ".jpg" << endl;

			}
			

		}
		
	}
	//_____________________________________________________________________________________________


	return 0;
}
