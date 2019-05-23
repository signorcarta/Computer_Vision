#include "panoramic_utils.h"
#include "PanoramicImage.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define FOCAL_LENGTH 18 //[mm]
#define vFOV 54 //[degrees] vertical Fiel of View
#define RATIO 3

int main(int argc, char** argv)
{
	std::vector<cv::Mat> imagesSet;
	cv::String path = "dataset_dolomites\\dolomites\\*.png";
	
	PanoramicImage panoramic(imagesSet, FOCAL_LENGTH, vFOV);
	panoramic.loadImages(path, panoramic);
    panoramic.showPanoramic(panoramic, RATIO, "dataset_dolomites\\panoramic.png");

	return 0;
}