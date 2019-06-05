#include "PlateDetection.h"

using namespace std;
using namespace cv;

void detectPlate(Mat& image, Mat& detected, String& path, int& platesFound, vector<Rect>& plates) {

	Mat src = image.clone();
	int n_platesFound;

	CascadeClassifier plateClassifier; ///Initializes a cascade classifier
	plateClassifier.load(path); ///Loads a pretrained model	
	plateClassifier.detectMultiScale(src, plates, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30)); ///Detect plates in the image
	n_platesFound = plates.size();
	
	
	///What to do if no plates is found (happens if is smaller than a certain size)______________________
	
	if (n_platesFound < 1) {		

		resize(src, src, Size(), 2.5, 2.5);
		plateClassifier.detectMultiScale(src, plates, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));
		n_platesFound = plates.size();
		
	}
	///___________________________________________________________________________________________________
	

	///What to do if several plates is found______________________________________________________________
	bool keepGoing = (n_platesFound > 1);

	if (n_platesFound > 1){
	
		int i = 0;

		while (keepGoing) {

			plateClassifier.detectMultiScale(src, plates, 1.1, 3+i, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));
			if (plates.size() == 1) { keepGoing = false; }
			else i++;

		}
		
	}
	///___________________________________________________________________________________________________
	
	/*
		I AM SO SURE I MANAGED TO FILTER JUST THE RIGHT LICENSE PLATE THAT I ACCESS DIRECTLY TO THE
		FIRST POSITION OF THE ARRAY WITHOUT CYCLING OVER THE VECTOR
	*/
	rectangle(src, plates[0], Scalar(0, 0, 255), 3, 8); ///Draws rectangle around the plate in the image
	

	detected = src;
	platesFound = plates.size();
	
}
