#include <vector>
#include <iostream>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "PanoramicUtils.h"
#include "PanoramicImage.h"

/// ////////////////////////////////////////////////////////////////////////// ///
/// Parameters have to be tuned according to which dataset one is going to use ///
/// ////////////////////////////////////////////////////////////////////////// ///

const double FOCAL_LEN = 18; //mm
const double vFOV = 54; //degrees
const double HvFOV = 27; // degrees
const double RATIO = 3;


using namespace std;
using namespace cv;



int main(int argc, char** argv) {
	
	int imgNum = 23;
	vector<Mat> imagesSet;
	
	PanoramicImage panor = PanoramicImage(imagesSet, FOCAL_LEN, vFOV); ///Instantiate an object 
	panor.loadImages("C:\\Users\\david\\source\\repos\\Keypoints_Descriptors_Matching\\dolomites\\", imgNum, panor); ///Load Images

	//Show and Save panoramic
	PanoramicImage::showAndSavePanoramic(panor, RATIO, "C:\\Users\\david\\source\\repos\\Keypoints_Descriptors_Matching\\panoramic.png");
	
	return 0;
}
