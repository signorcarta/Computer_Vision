#include "PlateDetection.h"

using namespace std;
using namespace cv;

void detectPlate(Mat& image, Mat& detected, String& path, int& platesFound, vector<Rect>& plates) {

	CascadeClassifier plateClassifier; ///Initializes a cascade classifier
	plateClassifier.load(path); ///Loads a pretrained model	
	plateClassifier.detectMultiScale(image, plates, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(20, 20)); ///Detect plates in the image

	for (size_t i = 0; i < plates.size(); i++) {
		
		rectangle(image, plates[i], Scalar(0, 0, 255), 3, 8); ///Draws rectangle(s) on the image
	}

	detected = image.clone();
	platesFound = plates.size();	
	
}
