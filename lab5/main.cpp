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

#define FOCAL_LEN 18 //mm
#define vFOV 54 //degrees
#define HvFOV 27 // degrees
#define RATIO 3 


using namespace std;
using namespace cv;



int main(int argc, char** argv) {
	
	// Loading
	int imgNum;
	vector<Mat> imagesSet;
	imagesSet = PanoramicImage::loadImages("C:\\Users\\david\\source\\repos\\Keypoints_Descriptors_Matching\\dolomites\\", imgNum);
	
	PanoramicImage pano = PanoramicImage(imagesSet, FOCAL_LEN, vFOV);

	//Show and Save panoramic
	PanoramicImage::showAndSavePanoramic(pano, RATIO, "C:\\Users\\david\\source\\repos\\Keypoints_Descriptors_Matching\\panoramic.png");
	
	return 0;
}
