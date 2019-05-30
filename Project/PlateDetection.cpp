#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>

#include "PlateDetection.h"

using namespace std;
using namespace cv;

void detectPlate(Mat& image, Mat& detected) {
	
	CascadeClassifier plateClassifier; ///Initialize a cascade classifier
	plateClassifier.load("classifier\\haarcascade_russian_plate_number.xml"); ///Load a pretrained model

	vector<Rect> plates; ///Vector of rectangles of the detected plates
	plateClassifier.detectMultiScale(image, plates);
	
	for (size_t i = 0; i < plates.size(); i++) {

		///Drawing rectangles on the image
		rectangle(image, plates[i], 255);

	}

	detected = image.clone();
	
}